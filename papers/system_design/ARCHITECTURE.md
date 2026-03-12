# Multi-Agent Imaging System Design Pipeline

A three-agent pipeline for physics-based imaging system design and reconstruction algorithm planning. Each agent is powered by Claude Opus 4.6 with adaptive thinking. The pipeline iterates through plan-judge-perform cycles to produce rigorous, physically grounded designs.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Agent Loop](#agent-loop)
4. [Plan Markdown Format](#plan-markdown-format)
5. [Directory Structure](#directory-structure)
6. [Module Reference](#module-reference)
   - [config.py](#configpy)
   - [main.py](#mainpy)
   - [agents/](#agents)
   - [database/](#database)
   - [schemas/](#schemas)
   - [pipeline/](#pipeline)
   - [utils/](#utils)
7. [Database Design](#database-design)
   - [Modality YAML Schema](#modality-yaml-schema)
   - [Algorithm Catalog Schema](#algorithm-catalog-schema)
   - [Adding a New Modality](#adding-a-new-modality)
8. [Flowchart Element Types](#flowchart-element-types)
9. [Noise Models](#noise-models)
10. [Mismatch Correction Registry](#mismatch-correction-registry)
11. [Reconstruction Algorithms](#reconstruction-algorithms)
12. [Usage](#usage)
13. [Examples](#examples)

---

## Overview

The system addresses two distinct design periods:

| Period | Goal | Plan Agent Output | Performance Agent Output |
|--------|------|-------------------|--------------------------|
| **Forward** | Design an imaging system | Flowchart of every physical element (source, medium, geometry, detector, ADC) with noise and mismatch at each stage | Simulated measurements `y` |
| **Reconstruction** | Design a reconstruction algorithm | Algorithm steps with equations, convergence criteria, and mismatch corrections | Reconstructed image `x_hat` + PSNR/SSIM |

Three agents collaborate:

- **Plan Agent** — reads the modality database and user prompt, generates a structured plan in markdown
- **Judge Agent** — validates physical/algorithmic feasibility before execution
- **Performance Agent** — executes the approved plan end-to-end in one pass

---

## Architecture

```
                          ┌────────────────────────────┐
                          │     YAML Database          │
                          │  modalities/ + algorithms/ │
                          └────────────┬───────────────┘
                                       │ context injection
                                       ▼
User Prompt ──► ┌─────────────┐   structured JSON   ┌─────────────┐
                │  Plan Agent  │ ──────────────────► │  plan.md     │
                │  (Opus 4.6)  │                     │  (4 sections)│
                └──────┬──────┘                      └──────┬──────┘
                       │                                    │
                       ▼                                    │
                ┌─────────────┐                             │
                │ Judge Agent  │ ◄──────────────────────────┘
                │  (Opus 4.6)  │    reads full plan markdown
                └──────┬──────┘
                       │
              ┌────────┴────────┐
              │                 │
          feasible=yes     feasible=no
              │                 │
              ▼                 ▼
   ┌──────────────────┐   redesign_prompt
   │ Performance Agent │   → back to Plan Agent
   │  Forward: simulate│   (max 3 iterations)
   │  Recon: reconstruct│
   └────────┬─────────┘
            │
            ▼
      results.npz
   (measurements or x_hat)
```

---

## Agent Loop

The orchestrator (`pipeline/orchestrator.py`) manages the cycle:

```
for iteration in 1..MAX_REDESIGN_ITERATIONS+1:
    1. Plan Agent generates plan (using DB context + optional judge feedback)
    2. Judge Agent evaluates the plan
       ├── PASS (feasible=true)  → proceed to step 3
       └── FAIL (feasible=false) → feed redesign_prompt back to step 1
    3. Performance Agent executes the plan
       ├── Forward:  simulate each flowchart element in topological order
       └── Recon:    run algorithm + mismatch corrections
    4. Return OrchestratorResult(plan, judgment, performance, success)
```

**Max redesign iterations**: 3 (configurable in `config.py`).
If the judge rejects the plan 3 times, the pipeline returns `success=False` with the last failure reason.

---

## Plan Markdown Format

Every plan is a structured markdown file with YAML front-matter and four required sections:

```markdown
---
modality: ct
period: forward
version: 1
iteration: 1
---

# Task
Design a sparse-view CT system with 60 projection angles for low-dose imaging.

# Plan
1. Configure X-ray tube source at 70 kVp with 2.5mm Al filtration
2. Model Beer-Lambert attenuation through soft tissue phantom
3. Define parallel-beam geometry with 60 projection angles
4. Simulate flat-panel detector with Poisson + Gaussian noise
5. Apply 14-bit ADC digitization

# Action
## System Flowchart
[Source] → [Phantom] → [Geometry] → [Detector] → [ADC] → y

### Element: X-ray Tube Source (`source`)
- **Type**: source
- **Parameters**: energy_kev=70, flux=1e6
- **Mismatch**: beam_hardening [high] → polynomial linearization

### Element: Flat Panel Detector (`detector`)
- **Noise**: poisson (I0=1e5), gaussian (sigma=5.0)
- **Mismatch**: detector_response_nonuniformity [low]
...

# Demands
- **feasibility**: yes
- **budget_feasible**: yes
- **algorithm_convergence**: N/A
```

### Section Details

| Section | Forward Period | Reconstruction Period |
|---------|---------------|----------------------|
| **1. Task** | Description of the imaging system to design | Description of the inverse problem to solve |
| **2. Plan** | Numbered steps referencing element names from the flowchart | Numbered steps referencing algorithm stages |
| **3. Action** | `ForwardAction` — ASCII flowchart + detailed element list with noise/mismatch per element | `ReconAction` — Algorithm steps with equations, mismatch corrections, hyperparameters |
| **4. Demands** | `feasibility` (yes/no), `budget_feasible` (yes/no), `comments` | `feasibility` (yes/no), `algorithm_convergence` (yes/no), `comments` |

---

## Directory Structure

```
papers/system_design/
├── __init__.py
├── config.py                        # Model IDs, max iterations, quality targets
├── main.py                          # CLI entry point
│
├── agents/
│   ├── __init__.py                  # Exports: PlanAgent, JudgeAgent, PerformanceAgent
│   ├── base_agent.py                # Claude API wrapper (streaming + JSON parsing)
│   ├── plan_agent.py                # Generates PlanDocument from prompt + DB context
│   ├── judge_agent.py               # Evaluates feasibility, returns JudgmentResult
│   └── performance_agent.py         # Executes approved plans (forward sim or recon)
│
├── database/
│   ├── __init__.py                  # Exports: get_modality, get_algorithms, list_modalities
│   ├── registry.py                  # YAML loader + platform DB fallback bridge
│   ├── modalities/
│   │   ├── ct.yaml                  # X-ray CT: source → attenuation → geometry → detector → ADC
│   │   ├── mri.yaml                 # MRI: B0 → RF → tissue → k-space → coil → ADC
│   │   └── widefield.yaml           # Widefield: LED → sample → objective → tube_lens → camera
│   └── algorithms/
│       └── catalog.yaml             # 22 algorithms across 7 categories
│
├── schemas/
│   ├── __init__.py                  # Exports all schema classes
│   ├── plan.py                      # PlanDocument, ForwardAction, ReconAction, FlowchartElement
│   └── judgment.py                  # JudgmentResult, JudgmentIssue
│
├── pipeline/
│   ├── __init__.py                  # Exports: Orchestrator, ForwardSimulator, Reconstructor
│   ├── orchestrator.py              # Plan → Judge → [redesign] → Perform loop
│   ├── forward_simulator.py         # Topological element-graph execution
│   └── reconstructor.py             # FBP, SART, TV-ADMM, PnP-ADMM dispatchers
│
├── utils/
│   ├── __init__.py
│   ├── noise.py                     # Poisson, Gaussian, dark current noise models
│   ├── mismatch.py                  # 8 registered mismatch correction functions
│   └── metrics.py                   # PSNR, SSIM
│
└── outputs/                         # Generated plan .md files and result .npz files
```

---

## Module Reference

### config.py

Central configuration for the entire pipeline.

| Setting | Default | Description |
|---------|---------|-------------|
| `PLAN_MODEL` | `claude-opus-4-6` | Model for plan generation |
| `JUDGE_MODEL` | `claude-opus-4-6` | Model for feasibility judgment |
| `PERF_MODEL` | `claude-opus-4-6` | Model for performance execution |
| `MAX_REDESIGN_ITERATIONS` | `3` | Max plan-judge-redesign loops |
| `PSNR_TARGET` | `40.0` dB | Default quality target |
| `SSIM_TARGET` | `0.90` | Default quality target |

---

### main.py

CLI entry point. Supports both invocation styles:

```bash
# From within system_design/
python3 main.py --modality ct --period forward --prompt "..."

# As a Python module from papers/
python3 -m system_design.main --modality ct --period forward --prompt "..."
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `--modality` | Yes | Modality key (e.g. `ct`, `mri`, `widefield`) |
| `--period` | Yes | `forward` or `reconstruction` |
| `--prompt` | Yes | Natural language design description |
| `--phantom` | No | Path to `.npy` ground truth (forward only) |
| `--measurements` | No | Path to `.npy` measurements (reconstruction, required) |
| `--x_true` | No | Path to `.npy` ground truth (reconstruction, for metrics) |
| `--no-verbose` | No | Suppress progress output |

---

### agents/

#### base_agent.py — `BaseAgent`

Thin wrapper around `anthropic.Anthropic()` with two main methods:

```python
class BaseAgent:
    def complete(self, system, messages, max_tokens=8192, thinking=True, stream_to_stdout=False) -> str
    def complete_json(self, system, messages, max_tokens=4096) -> dict
```

- Uses **streaming** to prevent HTTP timeout on long generations
- **Adaptive thinking** enabled by default (`thinking: {type: "adaptive"}`)
- `complete_json()` strips markdown code fences before JSON parsing

#### plan_agent.py — `PlanAgent(BaseAgent)`

```python
class PlanAgent:
    def generate(self, modality, period, prompt, judge_feedback="", iteration=1) -> PlanDocument
```

**How it works:**
1. Queries `database.registry` for the modality's forward elements and algorithm catalog
2. Injects this context into the user prompt alongside the user's design request
3. Sends to Claude with a system prompt tailored to the period (forward vs. reconstruction)
4. Parses the JSON response into a `PlanDocument` Pydantic model
5. Saves the rendered markdown to `outputs/{modality}_{period}_v1_iter{n}.md`
6. If `judge_feedback` is provided (from a prior rejection), includes it in the prompt

**System prompt templates:**
- `_SYSTEM_FORWARD` — instructs Claude to design complete forward models with every physical element, noise source, and mismatch
- `_SYSTEM_RECON` — instructs Claude to design algorithm steps with equations, convergence criteria, and mismatch corrections

#### judge_agent.py — `JudgeAgent(BaseAgent)`

```python
class JudgeAgent:
    def judge(self, plan: PlanDocument) -> JudgmentResult
```

**Judgment criteria by period:**

| Criterion | Forward | Reconstruction |
|-----------|---------|----------------|
| Physical correctness | Element parameters valid? | Steps mathematically consistent? |
| Noise / SNR | Estimated SNR > 15 dB? | N/A |
| Budget | Equipment cost realistic? | N/A |
| Convergence | N/A | Algorithm converges without pre-training? |
| Mismatch | All sources identified? | All critical sources corrected? |
| Completeness | All required elements present? | All steps described? |

**Failure output:** If `feasible=false`, the judge returns a specific `redesign_prompt` that the Plan Agent uses to fix the plan.

#### performance_agent.py — `PerformanceAgent`

```python
class PerformanceAgent:
    def run(self, plan, phantom=None, measurements=None, x_true=None) -> ForwardResult | ReconResult
```

**Forward period** (`ForwardResult`):
- Delegates to `ForwardSimulator.simulate()`
- Auto-generates a Shepp-Logan phantom if none provided
- Returns: `measurements` (ndarray), `ground_truth`, `element_outputs` (per-element debug arrays)

**Reconstruction period** (`ReconResult`):
- Delegates to `Reconstructor.reconstruct()`
- Computes PSNR/SSIM if `x_true` is provided
- Returns: `x_hat`, `psnr_db`, `ssim_val`, `convergence_history`, `runtime_s`

---

### database/

#### registry.py

Loads modality YAML files from `database/modalities/` and algorithm YAML from `database/algorithms/`. Falls back to the existing platform database (`platform/pwm_platform/services/modality_database.py`) for modalities not in the local YAML catalog.

**Key functions:**

| Function | Returns | Description |
|----------|---------|-------------|
| `get_modality(name)` | `dict` | Full modality config (YAML or platform fallback) |
| `get_algorithms(modality_name)` | `list[dict]` | Ordered algorithm list for a modality |
| `get_algorithm(alg_id)` | `dict` | Single algorithm entry |
| `list_modalities()` | `list[str]` | All known modality keys |
| `get_modality_context_for_prompt(name)` | `str` | Text summary for LLM prompt injection |
| `get_algorithm_context_for_prompt(name)` | `str` | Algorithm catalog text for LLM prompt |

All loaders use `@lru_cache` for performance.

#### modalities/*.yaml

Each YAML file defines a complete imaging modality. Three modalities are included:

**ct.yaml** — X-ray Computed Tomography:
```
Elements: X-ray Tube → Phantom Attenuation → Acquisition Geometry → Flat Panel Detector → ADC
Noise:    Poisson (I0=1e5) + Gaussian (σ=5e⁻) + Dark current (0.1 e⁻/s)
Mismatch: beam_hardening, scatter, center_of_rotation, detector_nonuniformity
Budget:   ~$500k equipment, ~$50/scan
```

**mri.yaml** — Magnetic Resonance Imaging:
```
Elements: Main Field (B0) → RF Excitation → Tissue Response → k-Space Sampling → Receive Coil → ADC
Noise:    Thermal (Johnson-Nyquist) + Physiological
Mismatch: B0/B1 inhomogeneity, chemical_shift, eddy_currents, motion, coil_sensitivity
Budget:   ~$2M equipment, ~$300/scan
```

**widefield.yaml** — Widefield Fluorescence Microscopy:
```
Elements: LED Excitation → Sample Fluorescence → Objective (60x/1.4NA) → Tube Lens + Dichroic → sCMOS Camera
Noise:    Poisson (shot) + Gaussian (readout 1.4e⁻) + Fixed pattern (0.5e⁻)
Mismatch: spherical_aberration, photobleaching, illumination_nonuniformity
Budget:   ~$80k equipment, ~$2/sample
```

#### algorithms/catalog.yaml

22 algorithms across 7 categories:

| Category | Algorithms | Applicability |
|----------|-----------|---------------|
| Classical | FBP, Zero-Filled IFFT, SENSE, GRAPPA, Richardson-Lucy, Wiener | CT, MRI, microscopy |
| Variational | SART, OSEM, TV-ADMM, L1-Wavelet, TV deconvolution, Blind deconvolution | CT, MRI, microscopy |
| PnP | PnP-ADMM (BM3D), PnP-DnCNN | CT, MRI, microscopy |
| Deep Unrolling | Unrolled ADMM, Deep ADMM-Net, E2E-VarNet | CT, MRI |
| Transformer | SwinMR | MRI |
| Diffusion | DiffusionMBIR | CT |
| Deep Learning | CARE, Deep Deconvolution (U-Net) | microscopy |

Each entry includes: `id`, `name`, `type`, `categories`, `parameters` (defaults), `reference` (published paper).

---

### schemas/

#### plan.py — Data Models

```
PlanDocument
├── modality: str
├── period: "forward" | "reconstruction"
├── version: int
├── iteration: int (redesign counter)
├── task: str                              # Section 1
├── plan_steps: list[str]                  # Section 2
├── action: ForwardAction | ReconAction    # Section 3
└── demands: PlanDemands                   # Section 4
```

**ForwardAction** (period = forward):
```
ForwardAction
├── flowchart_ascii: str          # ASCII diagram of signal path
├── elements: list[FlowchartElement]
│   └── FlowchartElement
│       ├── id: str
│       ├── name: str
│       ├── type: source | interaction | geometry | detector | digitization | processing | other
│       ├── parameters: dict
│       ├── noise: list[NoiseSpec]
│       │   └── NoiseSpec { type, parameters }
│       ├── mismatch: list[MismatchSpec]
│       │   └── MismatchSpec { type, description, severity, correction_method }
│       ├── connects_to: list[str]         # IDs of downstream elements
│       └── notes: str
├── measurement_shape: str
└── total_noise_model: str
```

**ReconAction** (period = reconstruction):
```
ReconAction
├── algorithm_name: str
├── algorithm_type: str
├── steps: list[AlgorithmStep]
│   └── AlgorithmStep { step, name, description, equation, parameters }
├── convergence_criterion: str
├── mismatch_corrections: list[MismatchSpec]
├── hyperparameters: dict
├── expected_runtime_s: float | None
└── references: list[str]
```

#### judgment.py — Verdict Models

```
JudgmentResult
├── feasible: bool
├── confidence: float [0.0, 1.0]
├── issues: list[JudgmentIssue]
│   └── JudgmentIssue { category, severity, element_id, description, suggestion }
├── summary: str
├── redesign_prompt: str                   # Fed back to Plan Agent if feasible=false
├── snr_estimate_db: float | None          # Forward only
├── cost_estimate_usd: float | None        # Forward only
├── budget_ok: bool | None                 # Forward only
├── convergence_likely: bool | None        # Reconstruction only
└── mismatch_handled: bool | None          # Reconstruction only
```

---

### pipeline/

#### orchestrator.py — `Orchestrator`

```python
class Orchestrator:
    def run(self, modality, period, prompt, phantom=None, measurements=None, x_true=None, verbose=True) -> OrchestratorResult
```

Returns `OrchestratorResult`:

| Field | Type | Description |
|-------|------|-------------|
| `plan` | `PlanDocument` | Final plan (last iteration) |
| `judgment` | `JudgmentResult` | Final judgment |
| `performance` | `ForwardResult \| ReconResult \| None` | Execution result (None if failed) |
| `iterations` | `int` | Number of plan-judge cycles |
| `success` | `bool` | Whether the pipeline completed successfully |
| `failure_reason` | `str` | Explanation if `success=False` |

#### forward_simulator.py — `ForwardSimulator`

```python
class ForwardSimulator:
    def simulate(self, action: ForwardAction, phantom: ndarray, modality: str) -> tuple[ndarray, dict[str, ndarray]]
```

**Execution model:** Topological sort of elements → process each in order.

Element type dispatch:

| Element Type | Handler | Physics |
|-------------|---------|---------|
| `source` | `_handle_source` | Generate incident flux from parameters |
| `interaction` | `_handle_interaction` | Beer-Lambert (CT), Bloch equations (MRI), multiplicative (generic) |
| `geometry` | `_handle_geometry` | Radon transform (CT), k-space undersampling (MRI) |
| `detector` | `_handle_detector` | Apply all noise models listed in `element.noise[]` |
| `digitization` | `_handle_digitization` | Quantize to ADC bit depth |
| `processing` | `_handle_processing` | PSF convolution (Gaussian approximation from NA + wavelength) |

The topological sort uses **Kahn's algorithm** on the `connects_to` graph, ensuring elements are processed in dependency order.

#### reconstructor.py — `Reconstructor`

```python
class Reconstructor:
    def reconstruct(self, action: ReconAction, measurements: ndarray, modality: str) -> tuple[ndarray, list[float]]
```

**Algorithm dispatch by `action.algorithm_type`:**

| Algorithm Type | Implementation | Notes |
|---------------|----------------|-------|
| Classical | FBP (`skimage.iradon`), Zero-filled IFFT, SART | Direct analytical or iterative |
| Variational / Compressed Sensing | TV-ADMM | Gradient descent + TV proximal |
| PnP / Plug-and-Play | PnP-ADMM | TV-ADMM init + Gaussian denoiser iterations |
| Deep Unrolling | Falls back to TV-ADMM | Requires pre-trained weights (not available) |
| Diffusion | Falls back to TV-ADMM | Requires pre-trained score model |

**TV-ADMM implementation:**
1. Initialize from FBP (sinogram) or zero-filled IFFT (k-space)
2. Compute data fidelity gradient (Radon domain or k-space domain)
3. Apply TV proximal operator (Gaussian smoothing approximation)
4. Non-negativity constraint
5. Convergence check: `||x_{k+1} - x_k|| / ||x_k|| < 1e-4`

All algorithms apply `mismatch_corrections` from the plan before reconstruction.

---

### utils/

#### noise.py — Physics-Based Noise Models

| Function | Model | Parameters |
|----------|-------|------------|
| `apply_poisson_noise(signal, I0)` | `y ~ Poisson(I0 * T)`, rescaled | `I0`: incident photon count |
| `apply_gaussian_noise(signal, sigma)` | `y = signal + N(0, σ²)` | `sigma`: std in signal units |
| `apply_dark_current(signal, rate, exposure_s)` | `y = signal + Poisson(rate * t)` | `rate`: e⁻/pixel/s; `exposure_s` |
| `add_mixed_poisson_gaussian(signal, I0, sigma)` | Poisson + Gaussian (CCD/sCMOS) | Combined detector model |

#### mismatch.py — Correction Registry

| Mismatch Type | Correction Method | Domain |
|---------------|-------------------|--------|
| `beam_hardening` | Polynomial linearization (2nd order) | CT |
| `scatter` | Subtract Gaussian-blurred low-frequency estimate | CT |
| `center_of_rotation_offset` | Cross-correlation alignment of 0°/180° projections | CT |
| `b0_inhomogeneity` | Phase ramp removal in k-space | MRI |
| `motion` | Gaussian windowing / navigator correction | MRI |
| `illumination_nonuniformity` | Flat-field correction | Microscopy |
| `photobleaching` | Exponential correction along time axis | Microscopy |
| `spherical_aberration` | Wiener deconvolution with Gaussian PSF model | Microscopy |

New corrections can be added by registering a function in `_CORRECTION_REGISTRY`.

#### metrics.py — Image Quality Metrics

| Function | Formula | Notes |
|----------|---------|-------|
| `psnr(x_hat, x_true)` | `10 * log10(range² / MSE)` | Returns `inf` for MSE=0 |
| `ssim(x_hat, x_true)` | Structural similarity (Wang et al. 2004) | Uses `skimage` with fallback |

---

## Database Design

### Modality YAML Schema

```yaml
name: <modality_key>                 # Used for lookup
display_name: "<Human Readable Name>"
aliases: ["<alt1>", "<alt2>"]        # Additional lookup keys
category: <tomography|mri|microscopy|spectroscopy|...>
physics_class: <x_ray_attenuation|nuclear_magnetic_resonance|fluorescence|...>
wave_model: <ray|electromagnetic_rf|incoherent|coherent>
sensor_type: <flat_panel_scintillator|rf_coil_array|scmos|...>
geometry: <circular_trajectory|k_space|planar|...>

forward_elements:
  - id: <snake_case_id>
    name: "<Human Name>"
    type: <source|interaction|geometry|detector|digitization|processing>
    parameters:
      <key>: <value>
    noise:
      - type: <poisson|gaussian|dark_current|thermal_johnson|...>
        parameters:
          <key>: <value>
    mismatch:
      - type: <mismatch_type>
        description: "<physical explanation>"
        severity: <low|medium|high>
        correction_method: "<how to correct>"
    connects_to: [<next_element_id>]

mismatch_sources:         # Summary list
  - <type1>
  - <type2>

algorithms:               # References to catalog.yaml IDs
  - <algorithm_id_1>
  - <algorithm_id_2>

budget_estimate:
  equipment_k_usd: <number>
  per_scan_usd: <number>

references:
  - "<citation>"
```

### Algorithm Catalog Schema

```yaml
algorithms:
  - id: <algorithm_key>
    name: "<Full Algorithm Name>"
    type: <Classical|Variational|PnP|Deep Unrolling|Diffusion|...>
    categories: [<modality_category>, ...]
    parameters:
      <key>: <default_value>
    runtime_complexity: "<big-O notation>"
    reference: "<published paper citation>"
    notes: "<additional context>"
```

### Adding a New Modality

1. Create `database/modalities/<name>.yaml` following the schema above
2. List the `forward_elements` in signal-flow order with `connects_to` edges
3. Specify noise models at the detector element
4. Specify all known mismatch sources with severity and correction methods
5. List algorithm IDs (must match entries in `algorithms/catalog.yaml`)
6. The registry auto-discovers the new file on next run

---

## Flowchart Element Types

| Type | Role | Example |
|------|------|---------|
| `source` | Generate incident radiation/field | X-ray tube, laser, RF coil, LED |
| `interaction` | Signal-specimen interaction | Beer-Lambert attenuation, Bloch equations, fluorescence emission |
| `geometry` | Spatial/frequency sampling | Radon projection, k-space trajectory, raster scan |
| `detector` | Signal detection + noise | CCD, sCMOS, flat panel, RF coil array |
| `digitization` | ADC / quantization | Bit depth, dynamic range |
| `processing` | Optical/electronic processing | PSF convolution, filtering, lens |
| `other` | Pass-through | Custom elements |

---

## Usage

### Forward Design

```bash
python3 main.py \
  --modality ct \
  --period forward \
  --prompt "Design a sparse-view CT system with only 60 projection angles, optimized for low-dose pediatric imaging"
```

**Output:**
- `outputs/ct_forward_v1_iter1.md` — the plan markdown (re-generated if judge rejects)
- `outputs/ct_forward_result.npz` — contains `measurements` and `ground_truth` arrays

### Reconstruction Design

```bash
python3 main.py \
  --modality mri \
  --period reconstruction \
  --prompt "Reconstruct 4x accelerated Cartesian k-space with retrospective motion correction using PnP-ADMM" \
  --measurements measurements.npy \
  --x_true ground_truth.npy
```

**Output:**
- `outputs/mri_reconstruction_v1_iter1.md` — the plan markdown
- `outputs/mri_reconstruction_result.npz` — contains `x_hat` and optionally `x_true`

### Programmatic Usage

```python
from system_design.pipeline import Orchestrator

orchestrator = Orchestrator()
result = orchestrator.run(
    modality="ct",
    period="forward",
    prompt="Design a low-dose CT with 90 angles and Poisson noise I0=1e4",
)

if result.success:
    y = result.performance.measurements      # sinogram
    plan_md = result.plan.to_markdown()       # full plan text
    print(f"Iterations: {result.iterations}")
    print(f"Judge confidence: {result.judgment.confidence:.2f}")
```

---

## Examples

### Example 1: Forward CT Design + Reconstruction

```python
from system_design.pipeline import Orchestrator
import numpy as np

orch = Orchestrator()

# Step 1: Design forward model
fwd = orch.run(modality="ct", period="forward",
    prompt="Sparse-view CT, 60 angles, low-dose (I0=1e4)")

# Step 2: Design reconstruction
if fwd.success:
    recon = orch.run(modality="ct", period="reconstruction",
        prompt="TV-ADMM reconstruction for sparse-view CT with beam hardening correction",
        measurements=fwd.performance.measurements,
        x_true=fwd.performance.ground_truth)

    if recon.success:
        print(f"PSNR: {recon.performance.psnr_db:.2f} dB")
        print(f"SSIM: {recon.performance.ssim_val:.4f}")
```

### Example 2: MRI with Motion Correction

```python
recon = orch.run(
    modality="mri",
    period="reconstruction",
    prompt="Reconstruct 4x accelerated k-space. The patient moved during the scan. "
           "Use ESPIRiT for coil calibration, then L1-wavelet CS with motion correction.",
    measurements=kspace_data,
    x_true=reference_image,
)
```

### Example 3: Adding a New Modality (OCT)

Create `database/modalities/oct.yaml`:

```yaml
name: oct
display_name: "Optical Coherence Tomography"
category: interferometry
physics_class: low_coherence_interferometry
forward_elements:
  - id: sld_source
    name: "Superluminescent Diode"
    type: source
    parameters: { center_wavelength_nm: 840, bandwidth_nm: 50 }
    connects_to: [interferometer]
  - id: interferometer
    name: "Michelson Interferometer"
    type: processing
    parameters: { reference_arm_length_mm: 5.0 }
    connects_to: [sample_interaction]
  - id: sample_interaction
    name: "Tissue Backscattering"
    type: interaction
    parameters: { model: born_approximation }
    mismatch:
      - type: dispersion_mismatch
        severity: medium
        correction_method: "Numerical dispersion compensation"
    connects_to: [spectrometer]
  - id: spectrometer
    name: "Spectrometer + Line Camera"
    type: detector
    noise:
      - type: poisson
        parameters: { I0: 5e4 }
    connects_to: [digitization]
  - id: digitization
    name: "ADC"
    type: digitization
    parameters: { bit_depth: 12 }
algorithms: [fbp, tv_admm, pnp_admm]
budget_estimate: { equipment_k_usd: 150, per_scan_usd: 20 }
```

Then run immediately — no code changes needed:

```bash
python3 main.py --modality oct --period forward --prompt "Design SD-OCT for retinal imaging"
```
