# Physics World Model (PWM)

The modality-agnostic **evaluation harness** and **current best reconstruction methods** for computational imaging — covering 64 modalities from microscopy to MRI to neural rendering.

---

## The Rail: Why the Harness Matters More Than Any Method

Think of PWM as a **railroad**, not a train.

The **harness** (OperatorGraph IR, 4-scenario protocol, Triad scoring, LIP-Arena) is the rail — durable infrastructure that defines *how* any method is tested, scored, and compared. The **methods** (GAP-TV, MST-L, HDNet, EfficientSCI, ...) are trains — they ride the rail, compete on it, and get replaced when something better comes along.

| Component | Role | Durability |
|-----------|------|------------|
| **Harness** (OperatorGraph IR, 4-scenario protocol, Triad scoring, LIP-Arena) | Defines *how* methods are tested | Durable — the railroad |
| **Current best methods** (43+ solvers across 64 modalities) | The methods that currently score highest | Replaceable — the trains |

One repo, one install — you get both. When your method scores higher on the harness, it becomes the new shipped default.

PWM implements the [SolveEverything.org](https://solveeverything.org/) 10-gear framework as a concrete reference trail. See [`rails/`](rails/README.md) for the complete gear mapping and [`docs/purpose.md`](docs/purpose.md) for the Imaging System Autonomy (ISA) discipline.

---

## Physics Fidelity Ladder

Every modality compiles into an **OperatorGraph**, and each node can run at a different physics tier. The same four-tier ladder spans all physical carriers — photons, electrons, spins, acoustic waves, particles.

| Tier | Label | Physics Regime | Example |
|------|-------|----------------|---------|
| 0 | `tier0_geometry` | Ray / ballistic — geometric optics, projection | CT projection, SEM beam geometry |
| 1 | `tier1_approx` | Wave / field approximations — Fourier, paraxial | Fresnel propagation, Bloch equations |
| 2 | `tier2_full` | Full transport / scattering — Maxwell, Monte Carlo | Full-wave EM, electron–matter scattering |
| 3 | `tier3_learned` | Learned surrogates with uncertainty | NeRF, diffusion priors (must provide error bars) |

**Key rule:** Tier is selected **per node**, not globally — a single graph can mix tiers. The graph compiler enforces a `TierPolicy` selecting the cheapest tier that meets accuracy and budget constraints.

**How it drives reconstruction:** correct the operator first (fit mismatch parameters via the Physics Fidelity Ladder), *then* reconstruct. This is why PWM's operator-correction mode consistently improves PSNR across all 16 validated calibration modalities.

See [`docs/operator_mode.md`](docs/operator_mode.md) for the full operator-correction pipeline.

---

## 4-Scenario Evaluation Protocol

Every validated modality is tested with 4 scenarios that isolate the effect of operator mismatch:

| Scenario | Measurement Operator | Reconstruction Operator | Purpose |
|----------|---------------------|------------------------|---------|
| I (Ideal) | True H | True H | Oracle upper bound |
| II (Assumed) | True H | Nominal H_nom | Mismatch impact baseline |
| III (Corrected) | True H | Calibrated H_hat | Calibration benefit |
| IV (Oracle) | True H | Partial oracle | Partial upper bound |

**Key metric:** Recovery ratio `rho = (PSNR_III - PSNR_II) / (PSNR_I - PSNR_II)` — how much of the oracle gap does calibration close?

See [`docs/targeting_system.md`](docs/targeting_system.md) for the full LIP-Arena specification.

---

## Quickstart

### Install

```bash
pip install -U pip
pip install -e packages/pwm_core
pip install -e packages/pwm_AI_Scientist   # optional: AI_Scientist adapter
pip install -e "packages/pwm_core[viewer]"  # optional: Streamlit viewer
```

### 1. Evaluate a method on the harness

```bash
# Score a method against the 4-scenario protocol
pwm evaluate --method my_solver --modality cassi --scenarios I,II,III,IV

# Quick sandbox evaluation
pwm evaluate --sandbox --modality widefield
```

### 2. Python API

```python
from pwm_core.api import endpoints

result = endpoints.run(prompt="widefield deconvolution, low dose")
print(f"PSNR: {result['recon'][0]['metrics'].get('psnr', 'N/A')}")
print(f"Verdict: {result['diagnosis']['verdict']}")
print(f"RunBundle: {result['runbundle_path']}")
```

### 3. Operator correction (measured data)

```python
from pwm_core.api import endpoints
from pwm_core.api.types import (
    ExperimentSpec, ExperimentInput, ExperimentStates,
    InputMode, PhysicsState, TaskState, TaskKind,
    MismatchSpec, MismatchFitOperator,
)

spec = ExperimentSpec(
    id="cassi_calib_001",
    input=ExperimentInput(mode=InputMode.measured, y_source="data/cassi_y.npy"),
    states=ExperimentStates(physics=PhysicsState(modality="cassi")),
    mismatch=MismatchSpec(enabled=True, fit_operator=MismatchFitOperator(
        enabled=True, search={"method": "random", "max_evals": 50},
    )),
)
result = endpoints.calibrate_recon(spec, out_dir="runs/")
print(f"Best-fit params: {result.calib.theta_best}")
```

See [`docs/quickstart/README.md`](docs/quickstart/README.md) and [`examples/`](examples/) for more.

---

## Repository Layout

```text
README.md
LICENSE
rails/                        # SolveEverything 10-gear implementation map
docs/                         # Architecture, specs, modality working-process docs
examples/                     # Runnable examples (prompt, calibration, operator correction)
community/                    # Weekly challenges, leaderboard, stewards
datasets/                     # InverseNet sample datasets (CASSI, CACTI, SPC)
packages/
  pwm_core/                   # Core library (operators, solvers, harness, agents)
    pwm_core/
      physics/                # 64 modality operators
      agents/                 # 17 agent modules
      analysis/               # Metrics, bottleneck, uncertainty
      core/                   # Runner, RunBundle, simulator
      api/                    # Pydantic types, endpoints
    contrib/                  # YAML registries (modalities, solvers, mismatch, photon, ...)
    benchmarks/               # 64-modality benchmark suite + 16 calibration tests
    tests/                    # Unit tests
  pwm_AI_Scientist/           # AI_Scientist adapter (thin wrapper)
CONTRIBUTING.md
```

---

## Community & Contributing

### 4 Levels of Contribution

| Level | What You Add | Difficulty | Example |
|-------|-------------|------------|---------|
| **Solver** | A new `ReconSolver` for an existing modality | Easy | Beat HDNet on CASSI |
| **Calibrator** | A new calibration method for operator correction | Medium | Better PSF estimator for lensless |
| **Modality** | A full modality (operator + CasePack + solver + tests) | Hard | Add STED microscopy |
| **Primitive** | A new OperatorGraph node type | Expert | New scattering kernel |

### Three-Speed Merge

| Lane | Scope | Timeline |
|------|-------|----------|
| **Fast** | Solvers, calibrators, config tweaks | Auto-merge within **48 hours** of CI pass (no human veto) |
| **Review** | Modalities, metrics, track tweaks | **7 days**, 2 reviewers required |
| **Governance** | Rail changes (scoring, protocol, frozen specs) | **90-day RFC**, unanimous steward vote |

### Weekly Challenges

Every week a new reconstruction challenge is posted in `community/challenges/`. Reconstruct from simulated measurements, submit a RunBundle, compete on the leaderboard. See [`community/CONTRIBUTING_CHALLENGE.md`](community/CONTRIBUTING_CHALLENGE.md).

For full contribution guidelines see [`CONTRIBUTING.md`](CONTRIBUTING.md). For governance details see [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md).

---

## Documentation Index

| Document | Description |
|----------|-------------|
| **Architecture** | |
| [`rails/README.md`](rails/README.md) | SolveEverything 10-gear framework + status table |
| [`docs/purpose.md`](docs/purpose.md) | Imaging System Autonomy (ISA) discipline |
| [`docs/spec_v0.2.1.md`](docs/spec_v0.2.1.md) | ExperimentSpec data model (8 state groups) |
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

---

## License

MIT. See [`LICENSE`](LICENSE).

## Citation

If you use PWM in academic work, please cite the associated paper (to be added) and link to this repository.
