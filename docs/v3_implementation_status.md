# PWM v3.0 Implementation Status

**Completed:** 2026-02-11
**Commits:** `99b78ff`, `3dc7f58`
**Test results:** 950 passed, 2 skipped, 0 failures

---

## What v3.0 delivered

The v3.0 milestone added the canonical chain structure (Source → Element(s) → Sensor → Noise)
to the graph IR, along with unified execution modes and supporting infrastructure.

### Files created (10)

| File | Lines | Purpose |
|------|-------|---------|
| `pwm_core/objectives/base.py` | 284 | NegLogLikelihood ABC + 6 impls + OBJECTIVE_REGISTRY |
| `pwm_core/objectives/prior.py` | 66 | PriorSpec (TV, L1-wavelet, low-rank, deep, L2, none) |
| `pwm_core/mismatch/belief_state.py` | 177 | BeliefState with bounds/prior/drift/history |
| `pwm_core/graph/executor.py` | 329 | GraphExecutor: unified Mode S/I/C |
| `pwm_core/graph/canonical.py` | 153 | Canonical chain validator |
| `pwm_core/api/prompt_parser.py` | 199 | Prompt → ParsedPrompt keyword parser |
| `tests/test_objectives.py` | 180 | NLL + prior tests (20 tests) |
| `tests/test_canonical_chain.py` | 363 | All-26 canonical validation + topology (16 tests) |
| `tests/test_graph_executor.py` | 252 | Mode S/I/C + operator correction (10 tests) |
| `tests/test_belief_state.py` | 113 | BeliefState CRUD + theta_space (10 tests) |

### Files modified (10)

| File | Changes |
|------|---------|
| `graph/ir_types.py` | PhysicsTier, NodeRole, CarrierType, DiffMode enums + NodeTags extension |
| `graph/primitives.py` | +12 primitives: Source×5, Sensor×4, SensorNoise×3 (40 total) |
| `graph/compiler.py` | Calls canonical validator when `metadata.canonical_chain = True` |
| `graph/graph_spec.py` | GraphNode gets `role: Optional[NodeRole]` |
| `graph/__init__.py` | Exports new modules |
| `contrib/graph_templates.yaml` | +26 v2 canonical templates (52 total) |
| `core/runner.py` | GraphExecutor integration path |
| `core/enums.py` | ExecutionMode (simulate/invert/calibrate) |
| `graph/source_spec.py` | SourceSpec, ExposureBudget, SpectrumSpec, CoherenceSpec |
| `graph/state_spec.py` | PhotonState, ElectronState, AcousticState, SpinState |

### Statistics

- **3,631 lines added** across 20 files
- **40 primitives** in registry (up from 28)
- **52 graph templates** (26 v1 + 26 v2)
- **56 new tests**, 950 total passing, 0 failures

---

## Known gaps (addressed in v3.1 roadmap → `plan_v3.md`)

1. All primitives are single-input (`forward(x)`); no multi-input fan-in support
2. Element nodes lack `physics_subrole`; Interaction/Transduction not enforced
3. NoiseModel and Objective are conflated via `_infer_objective_from_noise()`
4. Operator correction is an unstructured `provided_operator` swap, not a graph node
5. Canonical validator forces exactly-1 sensor; no multi-channel support
6. CT/Photoacoustic/NeRF/3DGS templates are physically inconsistent
7. `PhysicsTier` enum exists but is never populated or used for selection
8. `build_belief_from_graph` uses naive ±2x bounds, ignores `GraphNode.parameter_specs`
9. Prompt parser returns `ParsedPrompt` (plain class), not `ExperimentSpec`
10. `SourceSpec`, `ExposureBudget`, `StateSpec`, `PriorSpec` are orphaned (defined but unwired)
11. `GraphExecutor` is created in `runner.py` but never actually called (dead code)
12. `StrictBaseModel` is copy-pasted in 6+ files instead of being imported from one place
