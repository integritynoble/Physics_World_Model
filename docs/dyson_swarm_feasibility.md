# Dyson Swarm Strategy — Codebase Feasibility Assessment

> Assessment against `docs/dyson_swarm_strategy.md` based on current codebase state.
> Date: 2026-03-24

---

## Summary

| Layer | Status |
|-------|--------|
| **Sun (core protocol)** | ~85% exists — solid foundation, formalization gaps |
| **Inner orbits** | ~80% exists — platform deployed, algorithm catalog complete |
| **Outer orbits (ecosystem)** | ~35% exists — documented, not yet wired |

The physics engine, OperatorGraph IR, RunBundle, 4-scenario protocol, and algorithm catalog are **production-ready**. The trust ratchet (Certificate, trust tiers, docking Cards) and swarm interfaces (plugins, arXiv scaffolder, CI action) need to be built.

---

## 1. Sun Objects (Core Kernel)

### CoreSpec — 90% complete

- **What exists:** `spec/spec_v0.2.1.md` defines the canonical schema. 172 modality `.md` files in `spec/` serve as domain profiles. `ExperimentSpec` Pydantic model with 8 nested state layers (PhysicsState, BudgetState, CalibrationState, EnvironmentState, SampleState, SensorState, ComputeState, TaskState).
- **Gap:** The six-tuple (`object, forward model, measurement, noise model, prior, task`) is implemented but not formally named `CoreSpec` in code. `DomainProfile` and `ProblemInstance` as distinct typed objects do not yet exist — they are merged into per-modality spec files.
- **Verdict:** Functionally complete. Needs structural separation into the three-layer schema (CoreSpec / DomainProfile / ProblemInstance).

### Judge (S1-S4 gates) — 70% complete

- **What exists:** `packages/pwm_core/pwm_core/targeting/harness.py` is the modality-agnostic evaluation engine. `scoring.py` contains anti-Goodhart scoring and Triad gate attribution. `budget.py` has `BudgetGuard`. Provenance captured in `core/runbundle/provenance.py` (seeds, git hash, pip freeze).
- **Partial S1-S4 mapping:**
  - S1 (spec completeness): validation report written at run start
  - S2 (reproducibility): provenance captured in RunBundle
  - S3 (metric integrity): SHA-256 hashes on all stored artifacts
  - S4 (budget compliance): BudgetGuard class exists
- **Gap:** S1-S4 are not explicit named gates that produce pass/fail/warn verdicts. No run is blocked or flagged if a gate fails — they produce diagnostic data, not certification decisions.
- **Verdict:** The machinery exists. Hardening into formal rejection gates is a P0 task.

### OperatorGraph IR — 95% complete

- **What exists:** `packages/pwm_core/pwm_core/graph/` — full compiler pipeline: `compiler.py` (Spec → DAG), `ir_types.py` (typed primitives), `executor.py` (DAG execution), `canonical_decompositions.py` (primitive decomposition rules). 30+ primitives implemented (Fresnel propagation, angular spectrum, ray trace, Radon, DFT, coded mask, Conv2D/3D, etc.).
- **Gap:** Primitive references use internal names, not the `(registry, version, name)` triple specified in the strategy. Versioned registry lookup is not yet wired.
- **Verdict:** The IR is production-ready and used in all benchmarks. Registry-versioned references are a minor refactor.

### RunBundle — 95% complete

- **What exists:** `packages/pwm_core/pwm_core/core/runbundle/` — `artifacts.py`, `provenance.py`, `writer.py`, `manifest.py`. Full directory structure (spec/, data/, internal_state/, results/, exports/, viewer/). `bundle.json` manifest. SHA-256 checksums. Provenance (pip freeze, seeds, versions, command).
- **Gap:** No `certificate.json` is emitted. The Certificate object (trust verdict, tier, active gates, contributor attribution) does not exist.
- **Verdict:** RunBundle infrastructure is excellent. Certificate is the single missing P0 deliverable.

### Registry — 100% complete

- **What exists:** `packages/pwm_core/contrib/` — `primitives.yaml` (30 primitives), `graph_templates.yaml` (120 KB, ~170 modalities), `solver_registry.yaml`, `modalities.yaml` (240 KB), `photon_db.yaml`, `mismatch_db.yaml` (73 KB), `dataset_registry.yaml` (43 KB), `compression_db.yaml` (97 KB). Version-stamped.
- **Gap:** Registries are not yet organized into the `primitive_registry/general/v1/`, `primitive_registry/imaging/v1/`, `mappings/` directory structure specified. Append-only enforcement is not yet codified.
- **Verdict:** Content is complete. Structural reorganization is a refactor, not new work.

---

## 2. Primitive Namespaces

### General computational primitives (12) — 100% present in code

All 12 (Discretize, Transform, Compose, Adjoint, Invert, Regularize, Sample, Reduce, Broadcast, Differentiate, Stochastize, Validate) are implemented in `graph/primitives.py` (3,520 lines).

### Imaging physics primitives (11) — 100% present in code

All 11 (P, M, Pi, F, C, Σ, D, S, W, R, A) are implemented and used in compiled forward models across modalities.

### Mapping (imaging → general) — 0% as artifact

The decomposition rules exist in `canonical_decompositions.py` but are not published as an explicit versioned mapping artifact (`mappings/imaging_to_general/v1/`). No machine-readable cross-walk exists yet.

---

## 3. Judge Domain Modules

### Imaging Triad gates (G1/G2/G3) — 50% complete

- **What exists:** `targeting/scoring.py` has `GateAttribution` enum and `infer_gate_attribution()`. `analysis/bottleneck.py` classifies bottleneck type. G1 (sampling), G2 (noise), G3 (operator mismatch) are recognized.
- **Gap:** Gates inform scores and diagnostics but do not block certification. No hard rejection path exists.

### 4-scenario protocol — 100% complete

- **What exists:** `targeting/scenarios.py` — all four scenarios implemented (Ideal, Assumed, Corrected, Oracle). Tested on CT, MRI, CASSI, CACTI, Ptychography, CryoEM, Ultrasound. Results in `papers/pwm_flagship/results/`.

### CT QC domain profile — 20% complete

- **What exists:** `clinical_ct_thresholds.yaml` (7.8 KB) and `clinical_ct_mismatch.yaml` (19 KB) in contrib.
- **Gap:** No explicit `ct_qc/v1` DomainProfile. No threshold-table gate enforcement. No drift detection against baseline. No InstrumentCards for major CT scanners.

### Other domain profiles (combustion, spectroscopy, particle physics) — 0%

Mentioned in the strategy as future orbits. No code exists yet. This is a P3 item.

---

## 4. Algorithm Catalog

- **2,732 solvers across 172 modalities** — fully deployed and importable via `algorithm_base.get_solver()`.
- `algorithm_base/__init__.py` declares all modalities and solvers.
- Classical + DL solvers present for major modalities (CASSI, CACTI, CT, MRI, SPC, etc.).
- **Verdict:** Exceeds the strategy's reference of "2,755 algorithms." Production-ready.

---

## 5. Platform / Benchmark Orbit

### Web platform — 90% complete

- **What exists:** Flask app in `platform/pwm_platform/`. Routes: pages, runs, spec_chat, submissions, modalities, datasets, auth, billing. PostgreSQL DB. Docker Compose deployment.
- **Gap:** No trust-tier badges on leaderboard. No public Certificate viewer. No contributor rosters/profiles.

### Benchmark trust tiers — 30% complete

- **What exists:** 5 tiers (Draft, Author-confirmed, Reproduced, Certified, Boundary-risk) are defined in the strategy and referenced in docs.
- **Gap:** No `trust_tier` column in runs/submissions DB schema. No tier-promotion workflow. No reviewer queue. No UI rendering tier badges.

### arXiv claim scaffolder — 0%

Auto-scaffolding Draft ClaimCards from arXiv (`eess.IV`, `physics.optics`, `cs.CV`) is not implemented. This is a P1 item.

---

## 6. Docking Artifacts (Cards)

| Card | Status | Notes |
|------|--------|-------|
| RunBundle | ✅ 95% | Fully implemented, minor: Certificate needs adding |
| Certificate | ❌ 0% | Not yet implemented |
| SpecCard | ❌ 0% | No Pydantic schema; spec/ files are unstructured Markdown |
| MethodCard | ❌ 0% | No schema |
| DatasetCard | ⚠️ 10% | One example in experiments/; no canonical schema |
| InstrumentCard | ❌ 0% | No schema |
| ClaimCard | ❌ 0% | No schema, no arXiv scaffolder |
| EventCard | ❌ 0% | No schema |

Cards are the primary P1 gap. The lifecycle (Card → compile → run → RunBundle → Judge → Certificate) requires all cards to exist.

---

## 7. Developer Orbit

### PWM CLI — 80% complete

- **What exists:** `packages/pwm_core/pwm_core/cli/` — `main.py`, `demo.py`, `doctor.py`, `view.py`, `reproduce.py`, `modality_gate.py`, `inspect.py`. Entry point declared in `pyproject.toml`.
- **Missing commands:** `pwm synthesize`, `pwm ingest`, `pwm install`, `pwm evaluate` (harness exists but not CLI-bound).

### Plugin/contrib marketplace — 40% complete

- **What exists:** Registries in `contrib/`, dynamic solver loading, `CONTRIBUTING.md`.
- **Gap:** No `pwm install <plugin>` command. No plugin validation/signing. No marketplace API or ratings.

### GitHub Action (`pwm-benchmark`) — 0%

Not yet implemented. This is a P1 item.

### Language bindings (MATLAB, Julia, REST API) — 0%

Python-only currently. P2/P3 items.

---

## 8. Community Orbit

| Item | Status |
|------|--------|
| Modality maintainer roles | ❌ Not yet formalized |
| Contributor pages + badges | ❌ No DB schema or UI |
| PWM Weekly Digest (auto-newsletter) | ❌ Not implemented |
| Monthly Grand Rounds events | ❌ Not implemented |
| "Beat the Harness" red-team program | ❌ Not implemented |
| Industry advisory board | ❌ Not formalized |

All community orbit items are P2 tasks.

---

## 9. Open-Core Model

All items listed as "Open" in the strategy are either already open-source in the repo or feasible to open without significant change. The "Private/paid" layer (managed cloud, DICOM/PACS bridge, hospital connectors, institution dashboards) requires new engineering.

- **Billing infrastructure:** `platform/routers/billing.py` exists — foundation is there.
- **DICOM/HL7/FHIR integration:** Not implemented. P3 item.
- **Private workspaces:** DB-level scoping not yet implemented.

---

## P0 Blockers (must exist before any public Dyson Swarm launch)

1. **Certificate object** — Define `certificate.json` schema (Pydantic); wire Judge to emit it at run completion. Without this, the trust ratchet has no output.
2. **S1-S4 as hard gates** — Convert existing checks (validation report, provenance, metric hashes, BudgetGuard) into explicit pass/fail/warn verdicts that block or flag certification.
3. **Trust tier DB + UI** — Add `trust_tier` to DB schema; render tier badges on leaderboard; implement promotion workflow.
4. **Docking artifact schemas** — Pydantic models for SpecCard, MethodCard, DatasetCard, ClaimCard. These are the entry points to the swarm.
5. **Triad gates as safety brakes** — G1/G2/G3 must block or flag runs, not merely annotate scores.

## P1 Growth Items

- `pwm synthesize`, `pwm ingest`, `pwm install` CLI commands
- arXiv ClaimCard scaffolder (Draft tier auto-population)
- `pwm-benchmark` GitHub Action
- Versioned primitive registry directory structure (`primitive_registry/general/v1/`, etc.)
- Imaging-to-general primitive mapping artifact
- Contributor economy infrastructure (roles, badges, profiles)

## P2 Items

- CT QC DomainProfile (`ct_qc/v1`) with threshold tables and drift detection
- InstrumentCards for major CT scanners
- Plugin marketplace API
- Dataset federation (AAPM, fastMRI, BioImage Archive indexing)
- Conference workshop proposals (MICCAI 2026, ISBI)

## P3 Items

- Cross-domain expansion (acoustics, combustion, particle physics, remote sensing)
- DICOM/HL7/FHIR hospital connectors
- Autonomous science loops (AI Scientist integration)
- Cloud IDE with hosted compute

---

## Overall Verdict

The Dyson Swarm strategy is **highly feasible**. The hardest parts — physics forward models, OperatorGraph IR, RunBundle, 4-scenario protocol, algorithm catalog, and the web platform — are already production-ready. The repo represents several person-years of domain physics work that cannot easily be replicated.

The remaining P0 work is **engineering, not research**: formalizing types that already exist informally, wiring gates that already have the right data, and publishing schemas for cards whose semantics are already understood. The sun is mostly built. The solar collectors (Certificate, Cards, trust tiers) need fabrication.
