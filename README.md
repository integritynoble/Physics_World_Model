# PWM — Physics World Model for Computational Imaging

PWM is an open, reproducible **physics + reconstruction + diagnosis** toolkit for computational imaging.

It turns **either**:
- a **natural-language prompt** ("SIM live-cell, low dose, 9 frames...") **or**
- a structured **ExperimentSpec JSON/YAML** **or**
- **measured data** `y` + an imperfect operator/matrix `A` (**operator-correction mode**)

into a fully reproducible run:

**Prompt/Spec -> PhysicsState + WorldStates -> (Sim or Load) y -> (Fit operator theta) -> Reconstruct x_hat -> Diagnose -> Recommend actions -> RunBundle + Viewer**

PWM is designed to be:
- **Public and extensible** (plugins, CasePacks, dataset adapters)
- **Deterministic by default** (bounded search, reproducible seeds)
- **Embeddable** into agent systems like **Denario** (via `pwm_denario`)

---

## What PWM can do

### 1) Prompt-driven simulation + reconstruction

PWM supports **26 validated imaging modalities** with prompt-driven workflows:

**Microscopy:**
- `widefield` - Richardson-Lucy deconvolution (27.31 dB)
- `widefield_lowdose` - BM3D+RL for low photon counts (32.88 dB)
- `confocal_livecell` - Live-cell confocal with CARE (30.04 dB)
- `confocal_3d` - 3D stack with CARE 3D (39.17 dB)
- `sim` - Structured Illumination Microscopy, 2x resolution (27.48 dB)
- `lightsheet` - Stripe artifact removal (28.05 dB)

**Compressive Imaging:**
- `spc` - Single-Pixel Camera with LISTA (22.97 dB @ 25%)
- `cassi` - Hyperspectral imaging with MST-L (34.81 dB)
- `cacti` - Video snapshot compressive imaging with GAP-TV (26.88 dB)
- `lensless` - DiffuserCam with FlatNet (33.89 dB)

**Medical Imaging:**
- `ct` - Computed Tomography with RED-CNN (26.77 dB)
- `mri` - MRI with PnP-ADMM (44.97 dB)

**Coherent Imaging:**
- `ptychography` - Phase retrieval with Neural Network (59.41 dB)
- `holography` - Off-axis holography with Angular Spectrum (46.54 dB)
- `phase_retrieval` - CDI with Hybrid Input-Output (100.00 dB)
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

### 2) Operator correction mode (measured `y` + operator/matrix `A`)

For real experiments where the forward model is imperfect, PWM can:
- **fit/correct** forward-model parameters (theta) with a bounded calibration loop
- reconstruct with the corrected operator
- export a reproducible **RunBundle** including calibration trajectory and evidence

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

## Repository layout

```text
pwm/
  README.md
  LICENSE
  docs/
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
    pwm_core/              # public core library (no Denario deps)
      pwm_core/
        agents/            # 17 agent modules + contracts + registry
        physics/           # 26 modality operators
        analysis/          # Metrics, bottleneck, uncertainty
        core/              # Runner, RunBundle, simulator
        api/               # Pydantic types, endpoints
      contrib/
        modalities.yaml    # 26-modality source of truth
        mismatch_db.yaml   # Mismatch parameters per modality
        photon_db.yaml     # Photon models
        compression_db.yaml # Recoverability calibration tables
        metrics_db.yaml    # Per-modality metric sets
        solver_registry.yaml # 43+ solvers
      benchmarks/
        run_all.py         # 26-modality benchmark suite
        test_operator_correction.py  # 16 calibration tests
      tests/               # 570+ unit tests
    pwm_denario/           # Denario adapter (thin)
```

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
pip install -e packages/pwm_denario
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
1) select a CasePack from the 26 validated modalities,
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

# Option 1: Run from prompt (auto-selects casepack from 26 modalities)
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

# Run ALL 26 modalities (~28 min)
python benchmarks/run_all.py --all

# Run specific modality
python benchmarks/run_all.py --modality mri
python benchmarks/run_all.py --modality oct
python benchmarks/run_all.py --modality flim
python benchmarks/run_all.py --modality photoacoustic

# Run operator correction tests (16 tests, ~63 min)
python -m pytest benchmarks/test_operator_correction.py -v

# Run unit tests (570+ tests)
python -m pytest tests/ -v
```

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

This mode is for real experiments where the forward model is imperfect.

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

## 26 Imaging Modalities Benchmark

PWM includes validated implementations for **26 imaging modalities**.

### Summary

| # | Modality | Best Solver | PSNR (dB) | Ref (dB) | Status |
|---|----------|-------------|-----------|----------|--------|
| 1 | Widefield | Richardson-Lucy | 27.31 | 28.0 | Pass |
| 2 | Widefield Low-Dose | BM3D+RL | 32.88 | 30.0 | Pass |
| 3 | Confocal Live-Cell | CARE | 30.04 | 26.0 | Pass |
| 4 | Confocal 3D | CARE 3D | 39.17 | 26.0 | Pass |
| 5 | SIM | Wiener | 27.48 | 28.0 | Pass |
| 6 | CASSI | MST-L | 34.81 | 34.71 | Pass |
| 7 | SPC (25%) | LISTA | 22.97 | 32.0 | Warn |
| 8 | CACTI | GAP-Denoise | 26.88 | 26.5 | Pass |
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
| 21 | Phase Retrieval | HIO | 100.00 | 30.0 | Pass |
| 22 | FLIM | MLE Fit | 48.11 | 25.0 | Pass |
| 23 | Photoacoustic | Time Reversal | 50.54 | 32.0 | Pass |
| 24 | OCT | FFT Recon | 64.84 | 36.0 | Pass |
| 25 | FPM | Gradient Descent | 34.61 | 34.0 | Pass |
| 26 | DOT | Born/Tikhonov | 32.06 | 25.0 | Pass |

**25 of 26 modalities meet or exceed reference performance.** SPC at 25% below reference due to `deepinv` unavailable (basic FISTA fallback).

### Running the Benchmarks

```bash
cd packages/pwm_core

# Run ALL 26 modalities (~28 min)
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

# Unit tests (570+ tests)
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

## YAML Registries

PWM uses **6 YAML registries** as the source of truth for all modalities, solvers, and parameters:

| Registry | Entries | Purpose |
|----------|---------|---------|
| `modalities.yaml` | 26 modalities | Forward model families + upload templates |
| `mismatch_db.yaml` | Per-modality | Mismatch parameters and ranges |
| `photon_db.yaml` | Per-modality | Photon/noise models (model_id, not formulas) |
| `compression_db.yaml` | Calibration tables | Recoverability with provenance fields |
| `metrics_db.yaml` | Per-modality | Metric sets (phase_rmse, SAM, CNR, etc.) |
| `solver_registry.yaml` | 43+ solvers | Solver parameters and tier classification |

All registries are validated by Pydantic schemas with cross-reference integrity tests.

---

## Embedding into Denario

PWM exposes stable endpoints:
- `compile(prompt)` -> draft spec
- `resolve_validate(spec)` -> safe spec + auto-repair
- `simulate(spec)` / `reconstruct(spec, y)` / `analyze(...)`
- `fit_operator(...)` / `calibrate_recon(...)`
- `export(runbundle)` / `view(runbundle)`

Use `packages/pwm_denario/` as the thin adapter layer.

> You do **not** need AG2/LangGraph to run PWM.
> If you want autonomy loops later (planner<->reviewer, tool-using multi-step agents), implement them in `pwm_denario` without changing `pwm_core`.

---

## Contributing

PWM is intended to be extended by the community.

### Add a new modality/operator
1) Create a new operator in `pwm_core/physics/<modality>/`
2) Add YAML entries to all 6 registries (`modalities.yaml`, `mismatch_db.yaml`, `photon_db.yaml`, `compression_db.yaml`, `metrics_db.yaml`, `solver_registry.yaml`)
3) Add a benchmark in `benchmarks/run_all.py`
4) Add an operator correction test in `benchmarks/test_operator_correction.py`
5) Run `python -m pytest tests/test_registry_integrity.py` to verify no orphan keys

Templates:
- `pwm_core/contrib/templates/new_operator_template.py`
- `pwm_core/contrib/templates/new_calibrator_template.py`

### Add a dataset adapter
- Implement loader in `pwm_core/io/datasets.py` and format handler in `io/formats.py`
- Add an example under `examples/`
- Prefer reference-mode support for large datasets

---

## License

See `LICENSE`.

---

## Citation

If you use PWM in academic work, please cite the associated paper (to be added) and link to this repository.
