# PWM v3.1 Status — Verified Inventory

**Last verified:** 2026-02-11
**Branch:** `master` (commit `4fa488d`)
**Tests:** 1128 passing, 2 skipped, 0 failures (300s wall-clock)

---

## Core Graph Infrastructure

| Component | File | Lines | Key API | Tests |
|-----------|------|-------|---------|-------|
| IR types & enums | `graph/ir_types.py` | 250 | `NodeRole`, `PhysicsTier`, `PhysicsSubrole`, `CarrierType`, `CorrectionKind`, `DiffMode`, `NodeTags`, `TensorSpec`, `PortSpec`, `ParameterSpec`, `DriftModel` | `test_ir_types.py` |
| Graph compiler | `graph/compiler.py` | 305 | `GraphCompiler.compile(spec) -> GraphOperator` — validates DAG, binds primitives, topological sort, builds forward/adjoint plans. Shape inference from `metadata.x_shape`/`y_shape`. | `test_compiler.py`, `test_r6_templates.py` |
| Canonical chain validator | `graph/canonical.py` | 186 | `validate_canonical_chain(spec)` — enforces Source→Element(s)→Sensor→Noise, carrier-transition checking, at most 1 correction node, noise-is-sink | `test_canonical_chain.py` |
| Graph executor (Mode S/I/C) | `graph/executor.py` | 402 | `GraphExecutor`, `ExecutionConfig(mode, seed, solver_ids, ...)`, `ExecutionResult(y, x_recon, belief_state, metrics)`. Mode S = simulate, Mode I = invert (strip noise + solve), Mode C = calibrate (fit theta + optionally invert) | `test_graph_executor.py` |
| Graph adapter | `graph/adapter.py` | 127 | `GraphOperatorAdapter` — wraps `GraphOperator` into `BaseOperator` protocol. Noise stripping handled internally by `_StrippedGraphOp` in executor.py (not adapter). | `test_graph_adapter.py` |
| Graph spec | `graph/graph_spec.py` | — | `OperatorGraphSpec`, `GraphNode` (Pydantic, `extra="forbid"`) | Used by all template tests |
| Source spec | `graph/source_spec.py` | 238 | `SourceSpec`, `ExposureBudget` — carrier type, strength, spectral/spatial profile, coherence, dose constraints | `test_source_spec.py` |
| State spec | `graph/state_spec.py` | 184 | `PhotonState`, `ElectronState`, `AcousticState`, `SpinState` — carrier-specific state containers | `test_state_spec.py` |
| Tier policy | `graph/tier_policy.py` | 122 | `TierPolicy` — budget-based tier selection (tier0_geometry→tier3_learned) with per-modality overrides | `test_tier_policy.py` |
| Execution mode enum | `core/enums.py` | 23 | `ExecutionMode(simulate, invert, calibrate)` | Used by executor |

---

## Primitives (52 total in `graph/primitives.py`, 1686 lines)

### Source primitives (5) — `_node_role = "source"`

| Primitive | `primitive_id` | Carrier | Behavior |
|-----------|---------------|---------|----------|
| `PhotonSource` | `photon_source` | photon | `strength * x`, spatial profile support |
| `XRaySource` | `xray_source` | photon | photon source with keV spectrum hint |
| `AcousticSource` | `acoustic_source` | acoustic | acoustic carrier emission |
| `SpinSource` | `spin_source` | spin | RF excitation stub |
| `GenericSource` | `generic_source` | abstract | identity-like fallback |

### Sensor primitives (4) — `_node_role = "sensor"`

| Primitive | `primitive_id` | Carrier | Behavior |
|-----------|---------------|---------|----------|
| `PhotonSensor` | `photon_sensor` | photon | QE × gain + dark_current, multi-channel via `n_channels` param |
| `CoilSensor` | `coil_sensor` | spin | MRI coil sensitivity (complex multiply), multi-coil via `n_coils` |
| `TransducerSensor` | `transducer_sensor` | acoustic | acoustic-to-voltage conversion |
| `GenericSensor` | `generic_sensor` | abstract | identity with gain |

### Noise primitives (4 sensor + 4 legacy = 8)

**Sensor noise** (`_node_role = "noise"`, `_is_stochastic = True`):

| Primitive | `primitive_id` | Likelihood |
|-----------|---------------|-----------|
| `PoissonGaussianSensorNoise` | `poisson_gaussian_sensor` | Poisson shot + Gaussian read |
| `ComplexGaussianSensorNoise` | `complex_gaussian_sensor` | Complex Gaussian (MRI k-space) |
| `PoissonOnlySensorNoise` | `poisson_only_sensor` | Poisson-only |
| `GaussianSensorNoise` | `gaussian_sensor_noise` | Gaussian additive |

**Legacy noise** (no `_node_role`, backward compatibility):

| Primitive | `primitive_id` |
|-----------|---------------|
| `PoissonNoise` | `poisson_noise` |
| `GaussianNoise` | `gaussian_noise` |
| `PoissonGaussianNoise` | `poisson_gaussian_noise` |
| `FPN` | `fixed_pattern_noise` |

### Correction primitives (3) — `_node_role = "correction"`

| Primitive | `primitive_id` | Behavior |
|-----------|---------------|----------|
| `AffineCorrectionNode` | `affine_correction` | `gain * y + offset` (per-element) |
| `ResidualCorrectionNode` | `residual_correction` | `y + residual` (additive) |
| `FieldMapCorrectionNode` | `field_map_correction` | Multiplicative field map (e.g., MRI B0) |

### R6 Physically-correct primitives (5)

| Primitive | `primitive_id` | Subrole | Notes |
|-----------|---------------|---------|-------|
| `BeerLambert` | `beer_lambert` | transduction | `I_0 * exp(-sinogram)`, nonlinear, CT photon model |
| `OpticalAbsorption` | `optical_absorption` | interaction | `grueneisen * mu_a * x`, photon→acoustic carrier transition |
| `AcousticPropagation` | `acoustic_propagation` | propagation | Radon-like acoustic projection |
| `VolumeRenderingStub` | `volume_rendering_stub` | — | MIP rendering, tier3_learned |
| `GaussianSplattingStub` | `gaussian_splatting_stub` | — | Splatting + resize, tier3_learned |

### Other primitives (~27)

Transport/Propagation, PSF/Convolution (conv2d, conv3d), Modulation (coded_mask, sim_pattern), Warp/Dispersion (spectral_dispersion), Sampling (random_mask, temporal_mask, frame_integration), Nonlinearity (magnitude_sq), Fresnel propagation, Angular spectrum, CT Radon, MRI k-space, Identity, Multi-input, FourierRelay.

---

## Objectives & Noise Models

| Component | File | Lines | Contents |
|-----------|------|-------|----------|
| Objective ABC + 6 impls | `objectives/base.py` | 283 | `NegLogLikelihood` ABC, `PoissonNLL`, `GaussianNLL`, `ComplexGaussianNLL`, `MixedPoissonGaussianNLL`, `HuberNLL`, `TukeyBiweightNLL`, `OBJECTIVE_REGISTRY`, `build_objective()` |
| Noise models (4) | `objectives/noise_model.py` | 211 | `NoiseModel` ABC, `PoissonGaussianNoiseModel`, `GaussianNoiseModel`, `ComplexGaussianNoiseModel`, `PoissonOnlyNoiseModel`, `NOISE_MODEL_REGISTRY`, `noise_model_from_primitive()` |
| Prior spec | `objectives/prior.py` | — | `PriorSpec` (tv, l1_wavelet, low_rank, deep_prior, l2, none) |

---

## Mismatch & Calibration

| Component | File | Lines | Notes |
|-----------|------|-------|-------|
| BeliefState | `mismatch/belief_state.py` | 227 | `BeliefState`, `build_belief_from_graph()` — 3-tier priority (explicit ParamSpec > primitive defaults > generic) |
| Calibrators | `mismatch/calibrators.py` | — | `calibrate()`, `CalibConfig`, ThetaSpace integration |
| Scoring | `mismatch/scoring.py` | — | `poisson_nll`, `gaussian_nll`, `mixed_nll` |
| Subpixel shift | `mismatch/subpixel.py` | — | `subpixel_shift_2d()` — bilinear interpolation |
| Parameterizations | `mismatch/parameterizations.py` | — | `ThetaSpace` for calibration parameter management |

---

## API & Pipeline

| Component | File | Lines | Notes |
|-----------|------|-------|-------|
| ExperimentSpec v0.2.1 | `api/types.py` | — | Pydantic strict types, `extra="forbid"` |
| Endpoints | `api/endpoints.py` | — | `resolve_validate()`, `calibrate_recon()` |
| Prompt parser | `api/prompt_parser.py` | 309 | `parse_experiment_from_prompt()` — 26 modality keywords, mode detection, solver preferences |
| Runner | `core/runner.py` | — | Pipeline orchestrator, graph-first + parametric fallback |
| Physics factory | `core/physics_factory.py` | — | `build_operator()` — explicit spec > graph-first > modality routing. `_get_dims_from_spec()` handles dict format. |

---

## Registry Files

| File | Lines | Contents |
|------|-------|----------|
| `contrib/graph_templates.yaml` | 1925 | 50 templates (25 v1 + 25 v2), canonical chain enforced on v2 |
| `contrib/modalities.yaml` | 2300 | 25 modalities with keywords, categories, default solvers |
| `contrib/solver_registry.yaml` | 710 | 25 modalities × 4 tiers (traditional_cpu, best_quality, famous_dl, small_gpu) |
| `contrib/casepacks/` | 22 JSON + README | CasePack specs per modality |

---

## Test Suite

**Total:** 1128 tests passing, 2 skipped, 0 failures

| Test Area | File(s) | Count |
|-----------|---------|-------|
| R6 templates + primitives | `tests/test_r6_templates.py` | ~35 |
| CASSI theta-fit | `packages/pwm_core/tests/test_cassi_theta_fit.py` | 2 |
| Graph executor modes | `tests/test_graph_executor.py` | — |
| Canonical chain | `tests/test_canonical_chain.py` | — |
| BeliefState | `tests/test_belief_state.py` | — |
| Noise model | `tests/test_noise_model.py` | — |
| Tier policy | `tests/test_tier_policy.py` | — |
| Prompt parser | `tests/test_prompt_parser.py` | — |
| IR types | `tests/test_ir_types.py` | — |
| Correction nodes | `tests/test_correction_node.py` | — |
| Full suite (all areas) | `tests/` + `packages/pwm_core/tests/` | 1128 |

---

## Known Limitations (v3.1)

1. **No electron source/sensor primitives** — SEM/TEM/STEM/ET cannot be modeled
2. **No X-ray radiography, Ultrasound, PET/SPECT operators** — medical adoption blocked
3. **Correction nodes are element-wise only** — no pre/post transforms or low-rank operator updates
4. **No correlated noise model** — 1/f, spatial correlations not expressible
5. **TensorSpec not enforced at compile time** — shape errors caught at runtime only
6. **No CasePack acceptance tests** — "works well" not mechanically verified per modality
7. **Source-to-x wiring not explicitly enforced** — D2 not validated in canonical.py
8. **Canonical validator allows at most 1 correction node** — structured corrections may need >1
9. **Noise stripping in executor, not adapter** — `_StrippedGraphOp` is internal to executor.py
10. **No modality registry Python loader** — `modalities.yaml` exists but not programmatically queryable
