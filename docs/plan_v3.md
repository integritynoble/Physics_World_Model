# PWM v3 Plan: Canonical Source→Elements→Sensor→Noise Graph Structure

**Status: FULLY IMPLEMENTED** (2026-02-11)
**Commits:** `99b78ff`, `3dc7f58`
**Test results:** 950 passed, 2 skipped, 0 failures

---

## Context

PWM currently has a mature graph IR (OperatorGraphSpec, GraphCompiler, 30+ primitives, 26 templates) but the templates lack mandatory structure — some are just "blur + noise" with no source or sensor node. The user requires every modality to follow the canonical chain:

```
SourceNode → ElementNode(s) → SensorNode → NoiseNode → y
```

This enforces physical correctness (carrier propagation, exposure budget, likelihood-aware noise), enables unified execution modes (Simulate/Invert/Calibrate on the same graph), and supports both prompt-driven simulation and operator-correction mode.

**Scope:** ~3,500 lines of new/modified code across ~16 files
**Strategy:** Extend existing graph infra — don't rewrite what works

---

## File Layout

### New files
| # | File | Lines | Purpose | Status |
|---|------|-------|---------|--------|
| 1 | `pwm_core/objectives/base.py` | 245 | NegLogLikelihood ABC + 6 impls + OBJECTIVE_REGISTRY | DONE |
| 2 | `pwm_core/objectives/prior.py` | 70 | PriorSpec (TV, L1-wavelet, low-rank, deep, L2, none) | DONE |
| 3 | `pwm_core/mismatch/belief_state.py` | 170 | BeliefState dict with bounds/prior/drift/history | DONE |
| 4 | `pwm_core/graph/executor.py` | 260 | GraphExecutor: unified Mode S / I / C | DONE |
| 5 | `pwm_core/graph/canonical.py` | 130 | Canonical chain validator (called by compiler) | DONE |
| 6 | `pwm_core/api/prompt_parser.py` | 175 | Prompt → ExperimentSpec parser | DONE |
| 7 | `tests/test_objectives.py` | 120 | NLL + prior tests | DONE |
| 8 | `tests/test_canonical_chain.py` | 250 | All-26 canonical validation + topology tests | DONE |
| 9 | `tests/test_graph_executor.py` | 200 | Mode S/I/C + operator correction tests | DONE |
| 10 | `tests/test_belief_state.py` | 100 | BeliefState CRUD + theta_space conversion | DONE |

### Modified files
| # | File | Changes | Status |
|---|------|---------|--------|
| 1 | `graph/ir_types.py` | PhysicsTier, NodeRole, CarrierType, DiffMode enums + extended NodeTags | DONE |
| 2 | `graph/primitives.py` | +12 primitives: Source×5, Sensor×4, SensorNoise×3 (40 total) | DONE |
| 3 | `graph/compiler.py` | Calls canonical validator from compile() when flag set | DONE |
| 4 | `graph/graph_spec.py` | GraphNode gets `role` field (Optional[NodeRole]) | DONE |
| 5 | `graph/__init__.py` | Exports new modules (canonical, executor, NodeRole, PhysicsTier) | DONE |
| 6 | `contrib/graph_templates.yaml` | +26 v2 canonical templates (1,179 lines, 52 total) | DONE |
| 7 | `core/runner.py` | Integrates GraphExecutor path for canonical graphs | DONE |
| 8 | `core/enums.py` | ExecutionMode (simulate/invert/calibrate) | DONE |
| 9 | `graph/source_spec.py` | SourceSpec, ExposureBudget, SpectrumSpec, CoherenceSpec | DONE |
| 10 | `graph/state_spec.py` | PhotonState, ElectronState, AcousticState, SpinState | DONE |

---

## Milestones (dependency order)

### M0: Commit in-progress files — DONE
Committed the 4 uncommitted files (ir_types.py changes, enums.py, source_spec.py, state_spec.py) to lock the foundation.

### M1: objectives/base.py + prior.py (Spec §F2, §H) — DONE
**Created** `objectives/base.py`:
- `ObjectiveSpec(StrictBaseModel)` — kind + params
- `NegLogLikelihood(ABC)` — `__call__(y, yhat) -> float`, `gradient(y, yhat) -> ndarray`
- 6 concrete classes: `PoissonNLL`, `GaussianNLL`, `ComplexGaussianNLL`, `MixedPoissonGaussianNLL`, `HuberNLL`, `TukeyBiweightNLL`
- `OBJECTIVE_REGISTRY: Dict[str, type]` mapping kind strings to classes
- `build_objective(spec) -> NegLogLikelihood`

**Created** `objectives/prior.py`:
- `PriorSpec(StrictBaseModel)` — kind (tv/l1_wavelet/low_rank/deep_prior/l2/none), weight, params

**Reuses:** Existing NLL formulas from `mismatch/scoring.py` (poisson_nll, gaussian_nll, mixed_nll)

**Test:** `tests/test_objectives.py` — each NLL on synthetic data, gradient finite-diff check, build_objective round-trip, PriorSpec validation

### M2: New primitive families — Source, Sensor, Noise (Spec §E, §F) — DONE
**Modified** `graph/primitives.py` to add:

**SourcePrimitive family** (role=source, is_linear=True):
- `PhotonSource` — scales input by strength, applies spatial profile
- `XRaySource` — photon source with keV spectrum hint
- `AcousticSource` — acoustic carrier emission
- `SpinSource` — RF excitation stub
- `GenericSource` — identity-like fallback for Matrix/NeRF/3DGS
- Each stores a `SourceSpec` and `ExposureBudget` from params

**SensorPrimitive family** (role=sensor, is_linear=True):
- `PhotonSensor` — QE × gain + dark_current, outputs expected electron count
- `CoilSensor` — MRI coil sensitivity (complex multiply)
- `TransducerSensor` — acoustic-to-voltage conversion
- `GenericSensor` — identity with gain

**NoisePrimitive family** (role=noise, is_linear=False, is_stochastic=True):
- `PoissonGaussianSensorNoise` — Poisson shot + Gaussian read, **plus** `likelihood(y, y_clean) -> float`
- `ComplexGaussianSensorNoise` — for MRI k-space
- `PoissonOnlySensorNoise` — for CT
- Each noise primitive exposes `likelihood()` so the executor can infer the correct NLL for Mode I

**Registered** all in PRIMITIVE_REGISTRY (40 total primitives).

**Modified** `graph/graph_spec.py`: Added optional `role: Optional[NodeRole] = None` to `GraphNode`.

### M3: Canonical chain validator (Spec §A) — DONE
**Created** `graph/canonical.py`:
- `validate_canonical_chain(spec: OperatorGraphSpec) -> None`
  - Checks: exactly 1 source, ≥1 element, exactly 1 sensor, exactly 1 noise
  - Checks: directed path Source → ... → Sensor → Noise exists
  - Checks: Noise is a sink (no outgoing edges)
  - Role detection: first check `node.role`, then `node.tags.node_role`, then infer from `primitive_id` via PRIMITIVE_REGISTRY class attribute `_node_role`

**Modified** `graph/compiler.py`:
- In `compile()`, after `_validate_dag()` and `_validate_primitive_ids()`, calls `validate_canonical_chain(spec)` **only if** `spec.metadata.get("canonical_chain", False) is True` — preserves backward compat with v1 templates while enforcing the rule for v2+ templates

**Test:** `tests/test_canonical_chain.py` — valid chain passes, missing source/sensor/noise/elements fail, wrong topology fails, all 26 v2 templates compile

### M4: BeliefState (Spec §G) — DONE
**Created** `mismatch/belief_state.py`:
- `BeliefState(StrictBaseModel)`:
  - `params: Dict[str, ParameterSpec]` — mismatch param specs with bounds, prior, drift
  - `theta: Dict[str, float]` — current estimate
  - `uncertainty: Optional[Dict[str, float]]`
  - `history: List[Dict[str, float]]`
  - `update(new_theta, uncertainty=None)` — push current to history, set new
  - `get_bounds(name) -> (float, float)`
  - `to_theta_space() -> ThetaSpace` — bridge to existing calibrators
- `build_belief_from_graph(graph: GraphOperator) -> BeliefState` — extract learnable params

**Reuses:** Existing `mismatch/parameterizations.py` ThetaSpace, `mismatch/calibrators.py` CalibConfig

**Test:** `tests/test_belief_state.py` — creation, update history, bounds, theta_space conversion

### M5: GraphExecutor — unified Mode S / I / C (Spec §H, §L, §M) — DONE
**Created** `graph/executor.py`:
- `ExecutionConfig(dataclass)`:
  - `mode: ExecutionMode` (simulate/invert/calibrate)
  - `seed: int`, `add_noise: bool`
  - `solver_ids: List[str]`, `max_iter: int`
  - `objective_spec: Optional[ObjectiveSpec]`
  - `calibration_config: Optional[Dict]`
  - `provided_operator: Optional[PhysicsOperator]` — for operator-correction mode

- `ExecutionResult(dataclass)`:
  - `mode`, `y`, `x_recon`, `belief_state`, `metrics`, `diagnostics`

- `GraphExecutor`:
  - `__init__(graph: GraphOperator)` — stores graph, builds initial BeliefState
  - `execute(x=None, y=None, config=None) -> ExecutionResult` — dispatch by mode
  - **Mode S** `_simulate(x, config)`:
    1. Forward through all nodes except noise → y_clean
    2. Apply noise node with seed → y
    3. Return ExecutionResult(y=y, metadata={"y_clean": y_clean})
  - **Mode I** `_invert(y, config)`:
    1. Build forward op A (graph without noise, via _StrippedGraphOp)
    2. Infer objective from noise node type (or use config.objective_spec)
    3. Run solver portfolio → x_hat
    4. Return ExecutionResult(x_recon=x_hat, metrics=...)
  - **Mode C** `_calibrate(x, y, config)`:
    1. Build theta space from belief state
    2. Use existing `mismatch.calibrators.calibrate()` with theta space
    3. Update belief state with result
    4. Optionally run Mode I with calibrated params
    5. Return ExecutionResult(belief_state=updated, x_recon=...)
  - **Operator correction** — when `config.provided_operator` is set, use it instead of graph-derived A
  - `_infer_objective_from_noise() -> ObjectiveSpec` — inspect noise primitive_id → map to NLL kind

- `_StrippedGraphOp`: lightweight operator wrapping graph minus noise node for Mode I

**Test:** `tests/test_graph_executor.py`:
- Mode S: x → y with correct shape, noise present
- Mode I: y → x_hat with basic solver
- Operator correction: measured y + explicit operator A
- Objective inference from noise type

### M6: Rewrite 26 graph templates (Spec §K) — DONE
**Modified** `contrib/graph_templates.yaml` — every template has a v2 with canonical chain.

Pattern per modality:
```yaml
<modality>_graph_v2:
  metadata:
    modality: <key>
    canonical_chain: true
    x_shape: [H, W, ...]
    y_shape: [...]
  nodes:
    - node_id: source
      primitive_id: <carrier>_source
      role: source
      params: {strength: 1.0}
    - node_id: <element_1>
      primitive_id: <prim>
      role: transport
      params: {...}
      learnable: [...]
    - ...more elements...
    - node_id: sensor
      primitive_id: <carrier>_sensor
      role: sensor
      params: {quantum_efficiency: ..., gain: ...}
    - node_id: noise
      primitive_id: <noise_type>_sensor
      role: noise
      params: {peak_photons: ..., read_sigma: ..., seed: 0}
  edges:
    - {source: source, target: <element_1>}
    - ...chain...
    - {source: <last_element>, target: sensor}
    - {source: sensor, target: noise}
```

**26 modality mapping:**

| # | Modality | Source | Elements | Sensor | Noise |
|---|----------|--------|----------|--------|-------|
| 1 | Widefield | photon_source | conv2d (PSF) | photon_sensor | poisson_gaussian_sensor |
| 2 | Widefield Low-Dose | photon_source | conv2d (wider PSF) | photon_sensor | poisson_gaussian_sensor (high noise) |
| 3 | Confocal Live-Cell | photon_source | conv2d (tight PSF) | photon_sensor | poisson_gaussian_sensor |
| 4 | Confocal 3D | photon_source | conv3d | photon_sensor | poisson_gaussian_sensor |
| 5 | SIM | photon_source | sim_pattern → conv2d | photon_sensor | poisson_gaussian_sensor |
| 6 | CASSI | photon_source | coded_mask → spectral_dispersion → frame_integration | photon_sensor | poisson_gaussian_sensor |
| 7 | SPC (25%) | photon_source | random_mask | photon_sensor | poisson_gaussian_sensor |
| 8 | CACTI | photon_source | temporal_mask | photon_sensor | poisson_gaussian_sensor |
| 9 | Lensless | photon_source | conv2d (diffuser PSF) | photon_sensor | poisson_gaussian_sensor |
| 10 | Light-Sheet | photon_source | conv2d (sectioning) | photon_sensor | poisson_gaussian_sensor |
| 11 | CT | xray_source | ct_radon | photon_sensor | poisson_only_sensor |
| 12 | MRI | spin_source | mri_kspace | coil_sensor | complex_gaussian_sensor |
| 13 | Ptychography | photon_source | coded_mask → fresnel_prop → magnitude_sq | photon_sensor | poisson_gaussian_sensor |
| 14 | Holography | photon_source | fresnel_prop | photon_sensor | poisson_gaussian_sensor |
| 15 | NeRF | generic_source | ray_trace | generic_sensor | poisson_gaussian_sensor |
| 16 | 3D Gaussian Splatting | generic_source | ray_trace → conv2d | generic_sensor | poisson_gaussian_sensor |
| 17 | Matrix | generic_source | random_mask | generic_sensor | poisson_gaussian_sensor |
| 18 | Panorama Multifocal | photon_source | conv2d (defocus) → frame_integration | photon_sensor | poisson_gaussian_sensor |
| 19 | Light Field | photon_source | conv2d (microlens) | photon_sensor | poisson_gaussian_sensor |
| 20 | Integral | photon_source | conv2d (lenslet) | photon_sensor | poisson_gaussian_sensor |
| 21 | Phase Retrieval | photon_source | angular_spectrum → magnitude_sq | photon_sensor | poisson_gaussian_sensor |
| 22 | FLIM | photon_source | conv2d (PSF) → temporal_mask | photon_sensor | poisson_gaussian_sensor |
| 23 | Photoacoustic | photon_source | conv2d (absorption→acoustic) | transducer_sensor | poisson_gaussian_sensor |
| 24 | OCT | photon_source | angular_spectrum | photon_sensor | poisson_gaussian_sensor |
| 25 | FPM | photon_source | angular_spectrum → magnitude_sq | photon_sensor | poisson_gaussian_sensor |
| 26 | DOT | photon_source | conv2d (diffusion kernel) | photon_sensor | poisson_gaussian_sensor |

**Kept v1 templates** alongside v2 for backward compatibility (52 total templates).

### M7: Integrate into runner.py + prompt parser (Spec §H, §L, §M) — DONE
**Modified** `core/runner.py`:
- After building operator, if it's a GraphOperatorAdapter with `canonical_chain: true`, wraps in `GraphExecutor`
- Falls back to legacy path for non-canonical graphs

**Created** `api/prompt_parser.py`:
- `parse_prompt(prompt: str) -> ParsedPrompt` — keyword extraction:
  - Modality detection (26 keywords → modality_key)
  - Photon budget extraction ("low dose" → 1000, "bright" → 100000)
  - Mode detection ("simulate" → Mode S, "reconstruct" → Mode I, "calibrate" → Mode C)
  - Solver preference ("use FISTA" → solver_ids=["fista"])

### M8: Tests + verification — DONE
- `tests/test_objectives.py` — 20 tests: NLL correctness, gradient finite-diff check, registry lookup, PriorSpec
- `tests/test_canonical_chain.py` — 16 tests: all 26 v2 templates compile + validate, rejection tests, role inference
- `tests/test_graph_executor.py` — 10 tests: Mode S/I/C for widefield, operator correction, objective inference
- `tests/test_belief_state.py` — 10 tests: CRUD, theta_space bridge, build_belief_from_graph
- Full existing test suite: **950 passed, 2 skipped, 0 failures** (no regressions)

---

## Acceptance Criteria (mapped to user spec §A-§N)

| § | Requirement | Where | Verified by | Status |
|---|-------------|-------|-------------|--------|
| A | Mandatory chain: Source, Sensor, Noise, ≥1 Element | `canonical.py` validator | `test_canonical_chain.py` | PASS |
| A | Rejects violations | `canonical.py` raises `GraphCompilationError` | Rejection tests (6 cases) | PASS |
| B | Universal StateSpec | `state_spec.py` (PhotonState, ElectronState, AcousticState, SpinState) | Import tests | PASS |
| C | NodeSpec contract (ports, tags, mismatch, tiers) | `ir_types.py` NodeTags extended, GraphNode.role | All 26 templates compile | PASS |
| D | Physics Ladder (Tier 0-3) | `ir_types.py` PhysicsTier enum, primitives set tier | Template metadata | PASS |
| E | SourceSpec + ExposureBudget | `source_spec.py` | Source primitives use it | PASS |
| F | SensorNode + NoiseNode with likelihood | `primitives.py` new families | `test_graph_executor.py` sensor/noise tests | PASS |
| G | BeliefState | `belief_state.py` | `test_belief_state.py` | PASS |
| H | Mode S/I/C on same graph | `executor.py` GraphExecutor | `test_graph_executor.py` all 3 modes | PASS |
| I | YAML registries validated | `graph_templates.yaml` v2 + canonical validator | All 26 compile | PASS |
| J | RunBundle export | Existing (no changes needed) | Existing tests | PASS |
| K | 26 modality graphs | `graph_templates.yaml` v2 templates | `test_canonical_chain.py::test_all_26` | PASS |
| L | Acceptance criteria | All 26 compile, Mode S/I work, RunBundles export | Integration tests | PASS |
| — | Prompt-driven sim+recon | `prompt_parser.py` + executor | Unit test | PASS |
| — | Operator correction mode | `executor.py` provided_operator path | `test_graph_executor.py::test_operator_correction` | PASS |

---

## Implementation order (completed)

```
M0  Commit in-progress files                           ✓ 2026-02-11
 ↓
M1  objectives/base.py + prior.py                      ✓ 2026-02-11
M2  New primitives (Source/Sensor/Noise)                ✓ 2026-02-11
 ↓
M3  Canonical chain validator                           ✓ 2026-02-11
M4  BeliefState                                         ✓ 2026-02-11
 ↓
M5  GraphExecutor (S/I/C)                               ✓ 2026-02-11
M6  Rewrite 26 templates                               ✓ 2026-02-11
 ↓
M7  Runner integration + prompt parser                  ✓ 2026-02-11
M8  Tests + verification                                ✓ 2026-02-11
```

M1 and M2 ran in parallel. M3 and M4 ran in parallel. M5 and M6 ran in parallel.

---

## Final Statistics

- **3,631 lines added** across 17 files (10 new, 7 modified)
- **40 primitives** in registry (up from 28)
- **52 graph templates** (26 v1 + 26 v2)
- **56 new tests**, 950 total passing, 0 failures
- **2 commits**: `99b78ff` (foundation), `3dc7f58` (full implementation)
