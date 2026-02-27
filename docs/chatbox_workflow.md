# PWM Spec Builder — Chatbox Workflow

## Overview

The **Spec Builder** chatbox is the primary interface on the PWM dashboard (`/`). Users describe an imaging system in natural language, and the platform automatically designs a complete physics specification — including forward-model primitives, noise model, measurement matrix, and mismatch parameters — then simulates it with baseline solvers and ranks bottlenecks.

---

## End-to-End Flow

```
 User types natural language          HTMX POST
 ┌───────────────────────┐    ───►   /api/v1/spec-chat/{variant_key}
 │ "Design a CT scanner" │                    │
 └───────────────────────┘                    ▼
                                  ┌──────────────────────┐
                                  │  spec_chat.py        │
                                  │  (FastAPI router)    │
                                  │                      │
                                  │ 1. Create/retrieve   │
                                  │    SpecChatSession   │
                                  │    (PostgreSQL)      │
                                  │                      │
                                  │ 2. Append user turn  │
                                  │    to history        │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │  gemini_client.py    │
                                  │                      │
                                  │  Gemini 2.5 Flash    │
                                  │  via CompareGPT API  │
                                  │  temp=0.3, 4096 tok  │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │  Parse response:     │
                                  │  • explanation text   │
                                  │  • JSON spec block    │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │  Render HTMX partial │
                                  │  _spec_chat_message  │
                                  │  (user + assistant   │
                                  │   bubbles, DAG viz,  │
                                  │   "Run Simulation")  │
                                  └──────────┬───────────┘
                                             │
                                   User clicks "Run Simulation"
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │  spec_simulator.py   │
                                  │                      │
                                  │  Physics simulation  │
                                  │  → phantom → forward │
                                  │  → baselines → rank  │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │  Results rendered:   │
                                  │  GT | Meas | Recon   │
                                  │  PSNR/SSIM table     │
                                  │  Bottleneck chart    │
                                  │  Recommendations     │
                                  └──────────────────────┘
```

---

## Three-Step User Workflow

| Step | Action | What happens |
|------|--------|-------------|
| **1. Describe** | User types a system description (e.g. "Design a CASSI hyperspectral imager") | Gemini generates a complete spec with primitives, noise model, mismatch params |
| **2. Refine** | User requests changes (e.g. "increase noise level" or "add wavelength dispersion") | Multi-turn conversation; Gemini updates spec while preserving history |
| **3. Simulate** | User clicks **Run Simulation** | Physics engine runs forward model + baseline solvers, returns metrics & bottleneck analysis |

---

## Architecture Components

### Frontend

| File | Purpose |
|------|---------|
| `templates/dashboard.html` | Main page; includes chat box at line 63 |
| `templates/_spec_chat_box.html` | Chat widget: scrollable log, textarea input, Send button |
| `templates/_spec_chat_message.html` | Renders user/assistant bubbles, DAG pipeline, spec cards, "Run Simulation" button |
| `templates/_spec_simulation_result.html` | 3-panel image grid, method comparison table, bottleneck chart |

**Technology:** HTMX 2.0.4 (partial page updates, no full reloads) + Tailwind CSS.

### Backend Router — `routers/spec_chat.py`

Three endpoints:

```
POST /api/v1/spec-chat/{variant_key}          # Main chat (send message)
POST /api/v1/spec-chat/{variant_key}/example   # Load pre-built example
POST /api/v1/spec-chat/{variant_key}/simulate  # Run physics simulation
```

### LLM Client — `services/gemini_client.py`

| Setting | Value |
|---------|-------|
| Model | `gemini-2.5-flash` |
| API | CompareGPT OpenAI-compatible (`https://comparegpt.io/api`) |
| Temperature | 0.3 |
| Max tokens | 4096 |
| Timeout | 120 s |

### Session Persistence — `db/models.py`

```
SpecChatSession
├── session_id   (UUID, unique)
├── user_id      (optional — anonymous chat allowed)
├── variant_key  (which modality variant)
├── history      (JSONB array: [{role, content}, ...])
├── created_at
└── updated_at
```

In-memory cache provides fast reads; write-through to PostgreSQL on every turn.

---

## How Automatic System Design Works

### Step 1 — System Prompt Assembly (`services/spec_chat_prompts.py`)

The system prompt sent to Gemini is assembled from four components:

#### a) 11 Physics Primitives

| Symbol | Name | Example |
|--------|------|---------|
| **P** | Propagation | Free-space / medium kernels |
| **M** | Mask / Modulation | Coded apertures, SLM patterns |
| **Π** | Projection | Radon transform, fan/cone-beam |
| **F** | Fourier Sampling | MRI k-space, ptychography |
| **C** | Convolution | PSF (shift-invariant) |
| **Σ** | Summation / Integration | Spectral, temporal, angular collapse |
| **D** | Detector | Sensor readout + noise η |
| **S** | Structured Illumination | Hadamard, block, random patterns |
| **W** | Wavelength Dispersion | Prism, grating, shift α |
| **R** | Rotation / Motion | CT gantry, electron tilt |
| **Λ** | Wavelength Selection | Spectral filters |

#### b) 3 Reference Example Specs

Full worked examples for **CASSI**, **CACTI**, and **SPC-Block** — each with DAG, noise, measurement matrix, and mismatch table — so Gemini can learn the output format by example.

#### c) Category Context (22 categories)

Maps modality categories to typical DAG patterns:

| Category | Typical Pipeline |
|----------|-----------------|
| Compressive | `M → W → Σ → D` or `M → Σ → D` |
| Medical (CT) | `Π → D` |
| Microscopy | `C → D` or `C → S → D` |
| MRI | `F → D` |
| Coherent | `P + P → Σ → D` |
| Remote Sensing | `R → D` |

#### d) Carrier-Dependent Noise Models

Physical carrier determines default noise:

| Carrier | Default Noise |
|---------|---------------|
| Photon / X-ray | Mixed Poisson-Gaussian (η₄) |
| RF / Acoustic / Seismic | Gaussian (η₁) |
| Neutron / Proton / Muon | Poisson (shot noise) |

### Step 2 — LLM Generates Spec JSON

Gemini returns natural language explanation + a fenced JSON block:

```json
{
  "spec_notation": "Π(θ) → D(g, η₄)",
  "forward_model": [
    {"primitive": "Π", "params": "θ ∈ [0°, 180°], 180 views", "label": "Radon projection"},
    {"primitive": "D", "params": "512 detectors, η₄", "label": "Detector readout"}
  ],
  "noise_model": "Mixed Poisson-Gaussian: y = Poisson(Φx) + N(0, σ²)",
  "measurement_matrix": "Discrete Radon transform Φ ∈ R^{180×512 × 512×512}",
  "mismatch_params": [
    {"name": "angular_error", "symbol": "δθ", "description": "Gantry angle offset",
     "nominal": 0, "perturbed": 0.5}
  ]
}
```

### Step 3 — Spec Visualization

`_spec_chat_message.html` renders:
- **DAG pipeline**: color-coded primitive boxes connected by arrows
- **Noise model card**: carrier type and distribution
- **Measurement matrix card**: operator description
- **Mismatch table**: parameter name, symbol, nominal/perturbed values
- **"Run Simulation" button** (green, posts spec JSON to `/simulate`)

### Step 4 — Multi-Turn Refinement

The full conversation history is persisted in PostgreSQL and sent to Gemini on every turn. Users can iteratively refine:

- "Change the noise to pure Poisson"
- "Add a wavelength dispersion stage before the detector"
- "Increase the angular mismatch to 2 degrees"

Gemini maintains context and updates the spec accordingly.

---

## Physics Simulation Engine

### Dispatch Logic (`services/spec_simulator.py`)

```
variant_key
    │
    ├── Core InverseNet (8 variants)?  ──► Legacy paper results
    │   cassi, sd_cassi, dd_cassi,         (real KAIST/CACTI/Set11 data)
    │   cacti, spc, spc_block,
    │   spc_kronecker, matrix
    │
    └── All other modalities (160+)   ──► Category runner dispatch
                                           (synthetic physics simulation)
```

### 7 Category Runners (`services/category_runners/`)

| Runner | File | Modalities |
|--------|------|-----------|
| CT / Radon | `ct_radon.py` | CT, fan-beam, cone-beam |
| MRI k-space | `mri_kspace.py` | MRI, parallel MRI, compressed sensing MRI |
| Microscopy PSF | `microscopy_psf.py` | Widefield, confocal, two-photon, STED |
| Electron CTF | `electron_ctf.py` | SEM, TEM, STEM, electron tomography |
| Compressive Mask | `compressive_mask.py` | CASSI variants, SPC, coded aperture |
| Remote Sensing | `remote_sensing.py` | SAR, sonar |
| Scanning Probe | `scanning_probe.py` | AFM, STM |

Each runner implements the `CategoryRunner` interface:

```python
class CategoryRunner:
    def generate_phantom(config, rng) -> (array, name, colormap)
    def apply_forward_model(phantom, config, rng) -> (measurement, title, colormap)
    def get_baselines(config, rng) -> (methods, scenario_labels, dataset_label, attribution)
```

### Three-Scenario Evaluation

| Scenario | Description |
|----------|-------------|
| **I (Ideal)** | Perfectly calibrated operator — no mismatch |
| **II (Mismatched)** | Forward model has mismatch perturbations applied |
| **III (Oracle)** | Mismatch present but solver has oracle knowledge of it |

### Bottleneck Analysis

Four severity metrics (0–1 scale):

| Metric | Measures |
|--------|----------|
| **Photon / Noise** | Degradation from noise model |
| **Mismatch** | Gap between Scenario I and II |
| **Recoverability** | Absolute PSNR level achievable |
| **Solver Fit** | Gap between classical and deep methods |

Output:
```python
bottleneck = {
    "ranked": [
        {"factor": "mismatch", "severity": 0.65, "expected_gain_db": 8.2},
        {"factor": "noise_photon", "severity": 0.35, "expected_gain_db": 2.5},
    ],
    "what_to_change_first": "Improve aperture calibration (σ_shift ≤ 0.2px)"
}
```

---

## Simulation Result Structure

```python
@dataclass
class SimulationResult:
    ground_truth_path: str          # /static/simulations/{sim_id}/gt.png
    measurement_path: str           # /static/simulations/{sim_id}/measurement.png
    reconstructed_path: str         # /static/simulations/{sim_id}/recon.png
    psnr: float
    ssim: float
    solver_name: str
    methods: List[MethodResult]     # Multi-method comparison
    bottleneck: Dict                # Ranked severity + recommendations
    recommendations: List[str]      # "What to change first"
    dataset_name: str
    sim_id: str                     # UUID
    mismatch_gap_db: float          # Scenario I→II drop
    recovery_db: float              # Scenario II→III recovery
    modality: str
    scenario_labels: Dict[str, str] # I, II, III descriptions
    attribution: str                # Paper/source citation
    category_module: str            # Which category runner was used
    display_name: str
```

---

## 168 Imaging Modalities

| Source | Count | How |
|--------|-------|-----|
| `_variant_registry.py` | 65 | Hand-crafted specs with full DAG, noise, mismatch |
| `_modality_catalog.py` | 103 | Auto-expanded from catalog entries; inherit category context |
| **Total** | **168** | All receive simulation support via category runners |

**22 categories** spanning: Compressive, Medical, Microscopy, Electron, Coherent, Computational, Remote Sensing, Scanning Probe, and 14 more.

---

## File Map

```
platform/pwm_platform/
├── routers/
│   ├── spec_chat.py                     # Chat endpoints (3 routes)
│   └── pages.py                         # Dashboard page rendering
├── services/
│   ├── gemini_client.py                 # LLM client (Gemini 2.5 Flash)
│   ├── spec_chat_prompts.py             # System prompt assembly
│   ├── spec_simulator.py                # Simulation dispatch engine
│   ├── benchmark_database/
│   │   ├── _primitives.py               # 11 physics primitives (P, M, Π, ...)
│   │   ├── _variant_registry.py         # 65 core imaging variants
│   │   └── _modality_catalog.py         # 103 additional modalities
│   └── category_runners/
│       ├── _base.py                     # CategoryRunner interface
│       ├── ct_radon.py                  # CT reconstruction
│       ├── mri_kspace.py                # MRI k-space sampling
│       ├── microscopy_psf.py            # Microscopy PSF convolution
│       ├── electron_ctf.py              # Electron CTF phase contrast
│       ├── compressive_mask.py          # Compressive sensing masks
│       ├── remote_sensing.py            # SAR / sonar
│       └── scanning_probe.py            # AFM / STM
├── db/
│   ├── models.py                        # SpecChatSession ORM
│   └── database.py                      # SQLAlchemy async engine
├── templates/
│   ├── dashboard.html                   # Main page (includes chat)
│   ├── _spec_chat_box.html              # Chat widget UI
│   ├── _spec_chat_message.html          # Message + spec rendering
│   └── _spec_simulation_result.html     # Simulation results panel
└── main.py                              # FastAPI app entry point
```
