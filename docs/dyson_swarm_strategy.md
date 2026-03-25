# PWM as a Dyson Swarm: Sun + Orbits Architecture

> A Dyson Swarm works only if the sun is stable. PWM becomes a swarm not by
> building many gravity wells, but by hardening one small, trust-centered kernel
> and letting everything else orbit it. The sun is the protocol. The orbits are
> the domains, datasets, methods, instruments, events, and communities that dock
> to it. If the sun drifts, every collector drifts. If the sun holds, collectors
> can be added, replaced, or removed without system-level damage.

---

## 0. As-Built vs Target Architecture

PWM is not starting from zero. This strategy is a **formalization and hardening
plan over an already operational kernel**, not a greenfield design. Reading it
as a clean-sheet redesign would misrepresent the codebase and mislead
contributors about what actually needs to be built.

### Actual codebase topology (Windows public repo + Linux private/product repos)

PWM operates through **three distinct codebases** with different roles, access
controls, and deployment targets. They are not subfolders of one monorepo —
they are independent repositories that collaborate through explicit promotion.

#### Source-of-truth table

| Layer | Role | Local path | GitHub remote | Live URL |
|-------|------|-----------|---------------|----------|
| **Open protocol / schemas / CLI / benchmarks** | Public open-source kernel — canonical source of truth for all sun objects | `D:\onedrive\startup\program\physics_world_model\PWM5\pwm\public` (Windows) | `github.com/integritynoble/Physics_World_Model` | — |
| **Papers / private experiments / architecture notes** | Private research repo — prototypes ideas before promotion to open kernel | `/home/spiritai/pwm/Physics_World_Model/pwm` (Linux) | `github.com/integritynoble/pwm` | — |
| **Live platform / auth / billing / DB / deployment** | Private product repo — deploys the hosted platform | `/home/spiritai/pwm/Physics_World_Model/pwm_product` (Linux) | `github.com/integritynoble/pwm_product` | `https://pwm.platformai.org/` |

#### What each repo contains

**`Physics_World_Model` — open-source kernel** (Windows primary, Linux via clone)
```
algorithm_base/        2,732 solvers, 172 modalities
packages/pwm_core/     OperatorGraph, RunBundle, targeting, CLI
platform/              open web platform routes (reference implementation)
spec/                  172 modality domain profiles
benchmarks/, datasets/, docs/, tools/
```

**`pwm` — private research repo** (Linux)
```
papers/                InverseNet, CT QC Copilot, Finite Primitive Theorem, ...
notes/                 architecture notes, workspace setup docs
experiments/           private experiments and ablations
reports/               internal reports
```

**`pwm_product` — private product/deployment repo** (Linux → pwm.platformai.org)
```
platform/pwm_platform/ FastAPI app (auth, billing, DB, routers, templates)
platform/docker-compose.yml + Dockerfile
packages/pwm_core/     deployment copy of the core library (synced from public repo)
datasets/Benchmark/    benchmark datasets for live runs
configs/, deployment/  production configuration
```

#### Feature promotion ladder

Protocol and algorithm ideas flow through a one-way promotion pipeline:

```
1. Research stage   →  developed and prototyped in pwm
                        (papers, experiments, internal notes)
                                   ↓
2. Kernel stage     →  promoted into Physics_World_Model
                        (open protocol, open algorithms, open CLI)
                                   ↓
3. Product stage    →  synced into pwm_product/packages/pwm_core/
                        (deployment copy of stabilized kernel)
                                   ↓
4. Deployment stage →  released to https://pwm.platformai.org/
                        (docker compose up -d --build)
```

**Rule**: private research does not directly define the open protocol. A result
or algorithm must be promoted into `Physics_World_Model` before it becomes part
of the canonical sun. The live platform should consume stabilized kernel code,
not ad hoc research code.

#### Sync risk: deployment copy of core

`pwm_product/packages/pwm_core/` is a **deployment copy**, not the canonical
source. This creates a structural risk: if a kernel fix is applied only in
`pwm_product` and never promoted back to `Physics_World_Model`, the sun splits.

**Mitigation rules:**
- Core protocol changes (Certificate, gates, Card schemas, OperatorGraph) must
  be authored in `Physics_World_Model` first.
- `pwm_product` syncs from released/stabilized kernel state, not from ad hoc
  patches.
- Never hotfix protocol semantics only in `pwm_product`. If a fix is needed in
  production, it must be back-promoted to `Physics_World_Model` in the same cycle.

#### Repository governance principle

> **Protocol changes must land in `Physics_World_Model` first.**
> Private research (`pwm`) may prototype them. Product (`pwm_product`) may deploy
> them. But the sun must have one canonical public definition. If `Physics_World_Model`
> and `pwm_product` diverge on protocol semantics, the swarm loses its center of gravity.

### Current state summary

| Component | Status | Repo | Location |
|-----------|--------|------|----------|
| OperatorGraph IR (30+ primitives, typed DAG, compiler, executor) | **~95% built** | `Physics_World_Model` | `packages/pwm_core/pwm_core/graph/` |
| RunBundle (SHA-256 hashes, provenance, manifest, artifact storage) | **~95% built** | `Physics_World_Model` | `packages/pwm_core/pwm_core/core/runbundle/` |
| Algorithm catalog (2,732 solvers, 172 modalities) | **100% built** | `Physics_World_Model` | `algorithm_base/` |
| 4-scenario protocol (Ideal / Assumed / Corrected / Oracle) | **100% built** | `Physics_World_Model` | `packages/pwm_core/pwm_core/targeting/scenarios.py` |
| Registry (primitives, modalities, datasets, solver routing) | **~100% built** | `Physics_World_Model` | `packages/pwm_core/contrib/` |
| CoreSpec (`ExperimentSpec v0.2.1`, 172 modality domain profiles) | **~90% built** | `Physics_World_Model` | `spec/` |
| Live web platform (routes, DB, auth, billing, Docker) | **~90% built** | `pwm_product` | `platform/pwm_platform/` |
| Judge S1-S4 data (validation reports, provenance, hashes, BudgetGuard) | **data captured, gates not wired** | `Physics_World_Model` | `targeting/harness.py`, `budget.py` |
| Triad gates G1-G3 (bottleneck classification) | **scoring only, not safety brakes** | `Physics_World_Model` | `targeting/scoring.py`, `analysis/bottleneck.py` |
| Research papers (InverseNet, CT QC Copilot, Finite Primitive Theorem) | **in progress** | `pwm` | `papers/` |
| Certificate object | **not yet built** | `Physics_World_Model` | — |
| Benchmark trust tiers | **documented, not in DB or UI** | `pwm_product` | — |
| Docking artifact schemas (Cards) | **not yet built** | `Physics_World_Model` | — |
| CLI `pwm synthesize / ingest / install` | **not yet built** | `Physics_World_Model` | `packages/pwm_core/pwm_core/cli/` |

### Three build categories used throughout this document

- **[built]** — exists in production, may need minor cleanup or rename
- **[needs formalization]** — data or logic exists; needs to be wired, typed, or structured
- **[new build]** — does not exist; requires new engineering

The P0 gaps are nearly all in the second and third categories. The physics engine,
the evaluation machinery, and the platform are already there.

---

## 1. What the Sun Is

The sun is not a modality, not a benchmark, not a product. The sun is the
**minimal shared protocol** that makes scientific objects expressible, executable,
comparable, trustable, and reusable.

### Sun objects (the canonical kernel)

| Object | Role | Build state |
|--------|------|-------------|
| **CoreSpec** | Universal six-tuple problem description: (object, forward model, measurement, noise model, prior, task). Domain-agnostic. | [needs formalization] — exists as `ExperimentSpec v0.2.1` |
| **Judge** | Universal trust kernel. Checks structural invariants S1-S4 on every run. Domain modules extend but never bypass it. | [needs formalization] — data captured, gates not wired |
| **OperatorGraph** | Intermediate representation. The compiled forward model as a typed DAG of primitives. | [built] |
| **RunBundle** | Immutable audit record. Inputs, outputs, logs, metrics, seeds, hashes — everything needed to reproduce or dispute a result. | [built] |
| **Certificate** | Machine-readable trust verdict. Issued by the Judge after a RunBundle passes all gates. Carries a trust tier. | [new build] — the central P0 artifact |
| **Registry** | Versioned catalog of primitives, profiles, datasets, methods, instruments. The namespace authority. | [built] |

### Outer orbits (what docks to the sun)

| Orbit | Examples |
|-------|----------|
| Domain profiles | Computational imaging, CT QC, combustion CFD, spectroscopy, particle physics |
| Datasets | fastMRI, AAPM, BioImage Archive, synthetic phantoms, private clinical sets |
| Methods | GAP-TV, HDNet, diffusion solvers, classical regularizers, ViTs |
| Instruments | Siemens CT, Zeiss confocal, custom SIM rigs, radio telescopes |
| Events & claims | arXiv papers, conference results, clinical trial endpoints |
| Benchmarks | LIP-Arena tracks, modality leaderboards, QC drift reports |
| Community | Modality maintainers, dataset stewards, red-team contributors |

**Design rule**: Every **executable** scientific object eventually passes through
the Judge and can emit a RunBundle + Certificate. Non-executable cards (metadata-
only DatasetCards, unexecuted MethodCards, EventCards, early ClaimCards) enter
the Registry first and become Judge-eligible only once compiled into executable
PWM objects. No shortcut path exists for executable objects.

---

## 2. Kernel + Profiles Architecture

The old design risked a separate spec for every problem family. The new design
uses one kernel with layered specialization.

### The formula

```
CoreSpec + DomainProfile + ProblemInstance = executable PWM object
```

| Layer | What it is | Who writes it | How often it changes |
|-------|-----------|--------------|---------------------|
| **CoreSpec** | Universal six-tuple kernel. Defines object space, forward model signature, measurement space, noise model, prior, and task type (simulate / infer / calibrate / validate). | PWM core team | Rarely — semver-locked |
| **DomainProfile** | Domain-specific extension. For imaging: Triad gates, physics tier, 4-scenario protocol. For CT QC: threshold tables, artifact taxonomy, drift model. For combustion: reaction network, transport coefficients. | Domain maintainers | Per-domain cadence |
| **ProblemInstance** | Concrete case. Specific scanner, dataset split, patient, phantom, benchmark row. | End users / automation | Every run |

**Key constraint**: DomainProfiles extend CoreSpec — they add fields, they do not
override kernel fields. ProblemInstances bind concrete values to the combined
schema. The Judge validates all three layers.

### Migration from ExperimentSpec to CoreSpec

The codebase already has `ExperimentSpec v0.2.1` — an 8-layer Pydantic model
(`PhysicsState`, `BudgetState`, `CalibrationState`, `EnvironmentState`,
`SampleState`, `SensorState`, `ComputeState`, `TaskState`) — and 172 modality
`.md` files that serve as de facto DomainProfiles.

**Migration rules (no destructive rewrite):**

1. `ExperimentSpec` becomes the **implementation substrate** for `CoreSpec v1`.
   Introduce `CoreSpec` as a compatibility-preserving alias at P-1. Do not break
   existing callers.
2. Existing modality `.md` files become **DomainProfile assets**. They do not
   need to be rewritten — they need to be formally registered as DomainProfile
   instances with a versioned schema.
3. `ProblemInstance` is split out **gradually**, starting with the benchmark
   submission pipeline where per-run bindings are already explicit.
4. The existing `spec/spec_v0.2.1.md` is semver-locked and serves as the
   reference until `CoreSpec v1.0` is formally published.

**Anti-pattern to avoid**: do not begin a full schema rewrite before Certificate
and trust tiers are live. Terminology alignment is P-1; schema evolution is P1+.

---

## 3. Two Primitive Namespaces

The old strategy described "10 primitives." This conflated two levels of
abstraction that must be kept distinct.

### General computational primitives (12)

These are the universal building blocks for discretize / simulate / invert /
validate across any scientific domain:

| # | Primitive | Scope |
|---|-----------|-------|
| 1 | **Discretize** | Mesh, grid, or basis representation of continuous domain |
| 2 | **Transform** | Change of basis or coordinate system |
| 3 | **Compose** | Chain or fuse two operators into one |
| 4 | **Adjoint** | Formal transpose / backprojection |
| 5 | **Invert** | Direct or iterative solver step |
| 6 | **Regularize** | Prior / penalty / constraint injection |
| 7 | **Sample** | Sub-sampling, masking, or random access |
| 8 | **Reduce** | Summation, averaging, pooling over an axis |
| 9 | **Broadcast** | Replicate or tile across an axis |
| 10 | **Differentiate** | Gradient, Jacobian, or sensitivity map |
| 11 | **Stochastize** | Noise injection, Monte Carlo sampling |
| 12 | **Validate** | Metric computation, residual check, convergence test |

### Computational imaging physics primitives (11)

These are the physics-specific atoms for imaging forward models — the revised
Finite Primitive Basis:

| # | Primitive | Physics |
|---|-----------|---------|
| 1 | **Propagate (P)** | Free-space wave propagation |
| 2 | **Modulate (M)** | Element-wise field multiplication |
| 3 | **Project (Pi)** | Radon / line-integral projection |
| 4 | **Encode (F)** | Fourier-domain encoding (k-space) |
| 5 | **Convolve (C)** | Spatial convolution (PSF) |
| 6 | **Accumulate (Sigma)** | Summation over spectral / temporal axis |
| 7 | **Detect (D)** | Detector response (5 canonical families) |
| 8 | **Sample (S)** | Sub-sampling / undersampling |
| 9 | **Disperse (W)** | Wavelength-dependent spatial shift |
| 10 | **Scatter (R)** | Direction change and/or energy shift |
| 11 | **Attenuate (A)** | Beer-Lambert / exponential decay along path |

### Relationship

These are **complementary levels of abstraction, not rival primitive bases**.
The imaging primitives are a physics dialect; the general primitives are the
computational substrate. The sun owns the mapping:

```
PhysicsDialect → OperatorGraph → ComputePlan
```

Every imaging primitive decomposes into one or more general computational
primitives (e.g., Convolve = Transform + Compose + Transform^{-1}; Detect =
Reduce + Stochastize). Future domains (acoustics, particle physics, combustion)
define their own physics dialects, which map through the same OperatorGraph IR
to the same ComputePlan.

---

## 4. The Judge as Trust Kernel

The Judge is part of the sun, not a replaceable helper. Specialized domain
profiles do not bypass the core Judge — they extend it.

### Universal Judge kernel (S1-S4)

Every run, regardless of domain, must pass these structural checks:

| Gate | Check | Failure mode | Codebase asset |
|------|-------|-------------|----------------|
| **S1: Spec completeness** | All six CoreSpec fields are bound and type-valid | Reject: incomplete specification | `validation_report.json` at run start |
| **S2: Reproducibility** | RunBundle contains sufficient information to reproduce the result (seeds, versions, hashes) | Reject: non-reproducible | `core/runbundle/provenance.py` |
| **S3: Metric integrity** | Reported metrics match recomputed metrics from stored artifacts | Reject: metric tampering or drift | SHA-256 hashes in `artifacts.py` |
| **S4: Budget compliance** | Compute, memory, and wall-clock stayed within declared bounds | Warn or reject: budget overrun | `BudgetGuard` in `budget.py` |

> **Note on paper semantics vs operational gates**: The S1-S4 names above
> describe the operational gate family that will be live at launch. The formal
> paper semantics for these names may evolve as the scientific community aligns
> on terminology. The important invariant for the swarm launch is that the Judge
> emits explicit trust verdicts derived from already-captured evidence. Future
> work may tighten the alignment between these operational gate names and any
> published formal definitions.

### Domain Judge modules (additive, never subtractive)

| Domain | Additional gates | Build state |
|--------|-----------------|-------------|
| **Imaging** | Triad gates (G1: recoverability, G2: carrier budget, G3: operator mismatch), physics tier validation, 4-scenario consistency | [needs formalization] — scoring exists, hard gates not wired |
| **CT QC** | Threshold tables (noise, uniformity, CNR, MTF), artifact flags (ring, cupping, streak), drift detection against baseline | [new build] — threshold files exist, gate logic does not |
| **Combustion** | Mass/energy conservation residuals, reaction rate plausibility, mesh convergence | [new build] — P3 |
| **Spectroscopy** | Spectral resolution bounds, calibration lamp cross-check, baseline correction quality | [new build] — P3 |

**Rule**: A domain module can add gates. It cannot weaken or skip S1-S4.
A Certificate is issued only when all kernel gates AND all active domain gates
pass. The Certificate records which gates were active, so trust is interpretable.

### Certificate v1 — the central P0 artifact

Certificate is the most important missing object. Without it the trust ratchet
has no output: evidence is captured but never converted into a verdict that outer
orbits can dock to.

**Minimum Certificate v1 fields:**

| Field | Description |
|-------|-------------|
| `run_id` | Unique identifier linking to the RunBundle |
| `trust_tier` | One of: Draft / Author-confirmed / Reproduced / Certified |
| `risk_flags` | Set of overlay flags (see Section 7): boundary-risk, safety-brake, high-variance, reviewer-disputed |
| `active_gates` | List of gates that were evaluated (S1-S4 + domain gates) |
| `gate_verdicts` | Pass / fail / warn per gate, with short reason string |
| `triad_flags` | G1 / G2 / G3 attribution from Triad analysis |
| `provenance_hash` | SHA-256 of the RunBundle manifest |
| `contributor_attribution` | List of contributors credited for this run |
| `issued_at` | ISO 8601 timestamp |
| `judge_version` | Semver of the Judge kernel that issued this Certificate |

**Implementation target**: `packages/pwm_core/pwm_core/core/runbundle/certificate.py`

Certificate v1 should be **intentionally small**. Do not overdesign it before it
is emitted for real runs. The fields above are sufficient for the first golden
reference bundles and the first leaderboard trust-tier badges.

---

## 5. Canonical Object Model and Spec Hierarchy

### Spec hierarchy (not a monolithic file)

```
spec.core.md                    # CoreSpec — universal six-tuple
spec.<domain>.md                # DomainProfile — e.g., spec.imaging.md, spec.ct_qc.md
instance.yaml                   # ProblemInstance — concrete bindings

compiled artifacts:
  operatorgraph.json            # Compiled forward-model DAG
  computeplan.json              # Execution plan (device, budget, solver sequence)
  judge_report.json             # Full Judge output (all gates, pass/fail/warn)
  runbundle.json                # Immutable audit record
  certificate.json              # Trust verdict + tier + risk flags
```

`spec.core.md` is semver-locked and changes only through an RFC process.
`spec.<domain>.md` files are maintained by domain maintainers and versioned
independently. `instance.yaml` is ephemeral — one per run.

### Semantics-first, not parser-first

Markdown is the human face. Typed objects are the sun.

The canonical sun objects, in memory and on wire, are typed structures:

| Object | Format | Human view |
|--------|--------|-----------|
| `CoreSpec` | JSON Schema / Pydantic model | `spec.core.md` |
| `DomainProfile` | JSON Schema / Pydantic model | `spec.<domain>.md` |
| `PrimitiveDialect` | Typed registry entry | Docs page |
| `OperatorGraph` | JSON DAG | Interactive DAG viewer |
| `ComputePlan` | JSON execution tree | CLI `--dry-run` output |
| `RunBundle` | HDF5 + JSON manifest | Web viewer / `pwm view` |
| `TriadReport` | JSON structured diagnostic | Rendered report |
| `Certificate` | Signed JSON | Badge + detail page |

Markdown specs are rendered *from* these objects, not parsed *into* them.
The authoritative representation is always the typed object. Parsers are
convenience — they feed the compiler, which emits typed objects. If the
markdown and the object disagree, the object wins.

---

## 6. Registry and Translation Tables

### Versioned primitive registries

```
primitive_registry/
  general/v1/                   # 12 general computational primitives
  imaging/v1/                   # 11 imaging physics primitives
  mappings/
    imaging_to_general/v1/      # How each imaging primitive decomposes
```

All primitive content already exists in `contrib/primitives.yaml` and
`graph/primitives.py`. The P-1 task is a directory restructure and explicit
mapping artifact publication — not new primitive work.

### Protocol stability rules

1. **Registries are append-only within a major version.** A new primitive can be
   added to `general/v1` (it becomes `general/v1.x`). An existing primitive
   cannot be removed or redefined — that requires `general/v2`.

2. **Future domains add, never mutate.** A combustion team adds
   `primitive_registry/combustion/v1/` and
   `mappings/combustion_to_general/v1/`. They never touch `general/v1` or
   `imaging/v1`.

3. **Mappings are explicit and versioned.** The mapping from imaging primitives
   to general primitives is a concrete artifact, not implicit convention. It can
   be audited, tested, and evolved independently.

4. **The OperatorGraph IR is domain-neutral.** It references primitives by
   `(registry, version, name)` triples. The compiler resolves these at build
   time. This means an OperatorGraph from 2026 remains valid as long as its
   referenced registry version exists.

### Domain profile registries

```
domain_profiles/
  imaging/v1/                   # Triad gates, physics tiers, 4-scenario protocol
  ct_qc/v1/                     # Threshold tables, artifact taxonomy
  combustion/v1/                # Reaction networks, transport models
```

Same stability rules apply. A domain profile references a specific primitive
registry version, ensuring that compiled artifacts remain reproducible.

---

## 7. Orbits: Benchmark, Data, Developer, Community

These are the outer orbits — each one captures a different kind of value and
feeds it back through the sun.

### Benchmark orbit

**Primary beachhead: LIP-Arena for computational imaging.**

This is the first wedge. Not CT QC, not combustion, not a broad multi-domain
play. LIP-Arena already exists with 172 modalities and 2,732 algorithms. The
strategy is to harden it into the most trusted imaging benchmark in the field,
then use that credibility to expand.

#### Trust tiers and risk flags

Trust tiers are linear progression states. Risk flags are orthogonal overlays
that can coexist with any tier.

**Trust tiers** (every leaderboard row carries exactly one):

| Tier | Meaning | Who can promote |
|------|---------|----------------|
| **Draft** | Auto-scaffolded or self-reported. Not yet verified. | Automatic (system or self-submission) |
| **Author-confirmed** | Original authors have reviewed and accepted the PWM result. | Authors sign off on the RunBundle |
| **Reproduced** | At least one independent party has reproduced the result within declared tolerance. | Benchmark reviewer submits matching independent RunBundle |
| **Certified** | Passed full Judge (S1-S4 + domain gates) AND required evidence package complete. | Judge emits Certificate; requires both automated pass and reviewer sign-off |

**Risk flags** (overlays — do not replace tier, they annotate it):

| Flag | Meaning | Set by |
|------|---------|--------|
| `boundary-risk` | Metric variance overlaps with a rival method, or result is near a trust boundary | Automatic — Judge |
| `safety-brake` | A hard Triad gate (G1/G2/G3) failure was triggered | Automatic — Judge |
| `high-variance` | Metric spread across seeds or data splits exceeds declared epsilon | Automatic — Judge |
| `reviewer-disputed` | An independent reviewer has raised a technical objection | Manual — Benchmark reviewer |

A Certified row with a `boundary-risk` flag is still Certified — the flag means
"trust this result, but note the proximity to a decision boundary." A row cannot
reach Certified if a `safety-brake` flag is unresolved.

#### Trust-tier state machine

```
                 ┌──────────────────────────────────────────────┐
                 │                  RISK FLAGS                  │
                 │  (overlay any tier, set/cleared independently)│
                 └──────────────────────────────────────────────┘

[auto or self]         [author signs off]      [independent RunBundle]   [Judge pass + evidence]
    Draft       ──►    Author-confirmed   ──►       Reproduced       ──►      Certified
                                                                          (Certificate issued)

Demotion: any tier can be set back to Draft if the RunBundle is found to be
non-reproducible or if a gate that was previously waived is now enforced.
```

- **Draft → Author-confirmed**: Author submits acknowledgment linking their
  identity to the PWM RunBundle. No re-execution required.
- **Author-confirmed → Reproduced**: A benchmark reviewer (role defined in
  Section 9) runs an independent reproduction and submits a matching RunBundle
  within declared epsilon. System confirms match.
- **Reproduced → Certified**: Judge runs full S1-S4 + domain gates on the
  primary RunBundle. All gates pass. Certificate is emitted. Reviewer sign-off
  confirms evidence package is complete.

- **Conference integration**: Propose PWM as evaluation harness for MICCAI, ISBI,
  IEEE ICCP workshops. Offer blinded evaluation at zero cost. One major venue
  adopting PWM creates a standard-setting effect.

- **"Beat the Harness" challenge**: Standing open invitation — find a Triad gate
  the harness misdiagnoses, or break a safety brake, and earn red-team credit.

- **Controlled claim scaffolding**: New papers from arXiv (`eess.IV`,
  `physics.optics`, `cs.CV` imaging subset) are auto-scaffolded as **Draft**
  SpecCards. They are NOT auto-published as benchmark rows. They enter a review
  queue and advance through trust tiers only after author confirmation or
  independent reproduction. This prevents the leaderboard from becoming a noisy
  content farm.

### Data orbit

- **Federated dataset registry**: Index every public imaging dataset (AAPM,
  fastMRI, BioImage Archive, etc.) with standardized metadata: modality,
  primitive chain, resolution, noise model, license. PWM is the catalog, not
  necessarily the host.

- **Synthetic data factory**: One-click forward-model-based data generation via
  `pwm synthesize`. Fills gaps where real data is scarce or restricted.

- **"Bring Your Data" pipeline**: `pwm ingest <dir>` auto-detects modality,
  strips PHI, runs QC, emits a PWM-format DatasetCard.

### Developer orbit

- **Plugin marketplace**: Formalize `contrib/`. `pwm install <plugin>` pulls
  solvers, calibrators, dataset adapters. Ratings derived from LIP-Arena scores.

- **CI/CD integration**: GitHub Action `pwm-benchmark` runs the 4-scenario
  protocol on PRs. Once in CI, PWM is load-bearing infrastructure.

- **Language bindings**: Python-first, then MATLAB, Julia, REST API.

- **IDE integration**: VS Code extension for OperatorGraph visualization, inline
  Triad diagnostics, one-click benchmark submission.

### Community orbit *(P2+ — not launch blockers)*

> The items below are valuable orbit accelerators. None of them are part of the
> trust kernel, none are required for the first swarm release, and none should
> be prioritized over Certificate, hard gates, or trust tiers.

- **PWM Weekly Digest**: Auto-generated newsletter — new claims scaffolded,
  trust-tier promotions, leaderboard changes, upcoming events.

- **Monthly Imaging Grand Rounds**: Virtual seminar, recorded and published.

- **Hackathons**: "Close the oracle gap on modality X" competitions.

- **Industry advisory board**: Siemens Healthineers, GE Healthcare, Zeiss, Nikon,
  Canon Medical — early benchmark access in exchange for domain expertise.

---

## 8. Docking Artifacts

Every new paper, method, dataset, instrument, or claim enters PWM through a
minimal card-like artifact. Cards are lightweight, structured, and machine-
readable. They are the docking interface between the outer world and the sun.

### Minimum viable cards rollout

Do not build all cards at once. Build only the cards needed to support the
first trust ratchet. The rollout order is:

**P0 / P1 (required for first trust ratchet):**

| Card | Purpose | Key fields | Build state |
|------|---------|-----------|-------------|
| **Certificate** | Trust verdict from the Judge | run_id, tier, risk_flags, gate_verdicts, provenance_hash — see Section 4 | [new build] — P0 |
| **SpecCard** | Declares a problem to be solved | CoreSpec subset, DomainProfile ref, task type | [new build] — P1 |
| **MethodCard** | Declares a solver / algorithm | Name, version, code URI, primitive requirements, compute budget | [new build] — P1 |
| **DatasetCard** | Declares a dataset | Modality, size, license, noise model, split structure, provenance | [new build] — P1 |
| **ClaimCard** | Declares a result claim (from a paper, report, or experiment) | Source (DOI/arXiv), metric values, conditions, trust tier | [new build] — P1 |

**P2+ (after trust ratchet is live):**

| Card | Purpose | Key fields | Build state |
|------|---------|-----------|-------------|
| **InstrumentCard** | Declares a physical instrument | Manufacturer, model, primitive chain, calibration state | [new build] — P2 |
| **EventCard** | Declares a conference, workshop, challenge, or deadline | Date, venue, relevance, associated SpecCards | [new build] — P2 |

**Already built (sun objects, not cards):**

| Object | Build state |
|--------|-------------|
| **RunBundle** | [built] — `packages/pwm_core/pwm_core/core/runbundle/` |

**Lifecycle for executable cards**: Card → compile → run → RunBundle → Judge →
Certificate. A card that never compiles and runs stays at Draft tier. A card that
runs and passes the Judge earns a Certificate. This is the trust ratchet.

**Lifecycle for metadata-only cards**: Card → Registry entry → Judge-eligible
once compiled into an executable PWM object. EventCards and metadata-only
DatasetCards do not need to run to be useful — they are discovery artifacts, not
trust artifacts.

---

## 9. Contributor Economy *(P2+ — not launch blockers)*

> Contributor roles, badges, and credit surfaces are important for the swarm
> flywheel but are not required for the first public release. Implement after
> trust tiers and Certificate are live.

A Dyson Swarm works when every collector has identity and reward. PWM defines
explicit contributor roles and visible credit surfaces.

### Roles

| Role | Responsibility |
|------|---------------|
| **Modality maintainer** | Owns a modality's DomainProfile, curates its MethodCards and DatasetCards, reviews benchmark submissions |
| **Dataset steward** | Maintains dataset quality, versioning, and access. Responds to data issues |
| **Method integrator** | Adapts published algorithms into PWM-compatible MethodCards and solver plugins |
| **Judge-rule author** | Proposes and maintains domain Judge gates (Triad extensions, QC thresholds, etc.) |
| **Red-team contributor** | Probes the harness for failure modes, earns credit for discovered vulnerabilities |
| **Instrument contributor** | Provides InstrumentCards and calibration data for physical devices |
| **Claim curator** | Reviews auto-scaffolded ClaimCards, promotes or flags them through trust tiers |
| **Benchmark reviewer** | Independent reproduction of results; required for Reproduced tier |

### Credit surfaces

- **Contributor pages**: Public profile showing roles, contributions, trust-tier
  promotions facilitated.
- **Badges**: Earned for milestones (first Certified result, 10 reproductions,
  red-team find, etc.).
- **Citation credit**: RunBundles and Certificates carry contributor attribution.
  When a RunBundle is cited in a paper, contributors are credited.
- **Maintainer rosters**: Each modality page lists its maintainers prominently.
- **Challenge credits**: Hackathon and "Beat the Harness" winners displayed on
  leaderboards.

---

## 10. Three Flywheels

The old strategy had one giant flywheel mixing everything. These are three
distinct loops with different speeds, participants, and value types.

### Knowledge flywheel (slow, durable)

```
Interactive modality pages → Unified glossary → Curriculum paths
        ↑                                              ↓
  SEO / discoverability ← Student + researcher discovery
```

- **Fuel**: Every new DomainProfile generates explainer pages automatically —
  OperatorGraph DAG, primitive decomposition, Triad failure sliders, top
  algorithms with metric curves.
- **Speed**: Months. Content compounds over time.
- **Metric**: Organic search traffic, curriculum completion rate, glossary usage.

### Research flywheel (medium, citation-driven)

```
ClaimCard ingestion → SpecCard generation → Benchmark execution
        ↑                                          ↓
  Method submissions ← Citations ← Trust-tier promotion
```

- **Fuel**: Every new paper produces a ClaimCard. Confirmed claims become
  benchmark rows. Benchmark rows get cited. Citations attract more submissions.
- **Speed**: Weeks to months per cycle.
- **Metric**: Trust-tier promotion rate, citation count, method submission rate.

### Operations flywheel (fast, revenue-driven)

```
Instrument onboarding → QC runs → Drift detection → Reports
        ↑                                              ↓
  Enterprise contracts ← Recurring value ← Compliance evidence
```

- **Fuel**: Each deployed instrument (starting with CT scanners) generates
  recurring QC runs. Drift detection triggers re-evaluation. Reports satisfy
  compliance workflows. Recurring value justifies enterprise contracts.
- **Speed**: Days to weeks per cycle.
- **Metric**: Instruments onboarded, QC runs per month, drift alerts, contract
  renewals.

---

## 11. Implementation Phases

> **Reading guide for contributors:** Each deliverable is tagged `[built]`,
> `[needs formalization]`, or `[new build]`. Each phase ends with explicit exit
> criteria — the definition of done. Implementation targets name the real
> packages and files.

---

### P-1 — Align terminology *(prerequisite, no new functionality)*

The codebase is more advanced than a clean read of this document suggests.
P-1 is purely alignment: rename, restructure, and alias existing code to match
the canonical vocabulary. No new features. No breaking changes.

| Deliverable | Current state | Action | Repo | Target |
|------------|--------------|--------|------|--------|
| **CoreSpec alias** | `ExperimentSpec v0.2.1` | Introduce `CoreSpec` as compatibility-preserving alias; begin splitting into CoreSpec / DomainProfile / ProblemInstance layers | `Physics_World_Model` | `packages/pwm_core/pwm_core/spec/core.py` |
| **Primitive registry layout** | `contrib/primitives.yaml` + `graph/primitives.py` | Reorganize into `primitive_registry/general/v1/`, `imaging/v1/`, `mappings/imaging_to_general/v1/` | `Physics_World_Model` | `packages/pwm_core/contrib/primitive_registry/` |
| **OperatorGraph primitive refs** | Internal string names | Switch to `(registry, version, name)` triples | `Physics_World_Model` | `packages/pwm_core/pwm_core/graph/ir_types.py` |

**Exit criteria:**
- `from pwm_core.spec import CoreSpec` works; existing `ExperimentSpec` callers unbroken
- Primitive registry files exist at target paths; old paths symlinked or aliased
- CI passes with no regressions

---

### P0 — Complete the sun *(5 genuine gaps)*

The OperatorGraph IR, RunBundle, 4-scenario protocol, algorithm catalog (2,732
solvers, 172 modalities), and platform are already production-ready. P0 addresses
only the five things that are genuinely missing from the trust kernel.

| Deliverable | State | Repo | Target package / file | Description |
|------------|-------|------|----------------------|-------------|
| **Certificate object** | [new build] | `Physics_World_Model` | `packages/pwm_core/pwm_core/core/runbundle/certificate.py` | Pydantic model + `certificate.json` emitter. See Section 4 for required fields. Wire into `harness.py` so every completed run emits one. Then sync to `pwm_product`. |
| **S1-S4 as hard gates** | [needs formalization] | `Physics_World_Model` | `packages/pwm_core/pwm_core/targeting/harness.py`, `budget.py`, `provenance.py` | Convert existing data captures (validation report, provenance, SHA-256 hashes, BudgetGuard) into explicit pass/fail/warn verdicts. Failed S1/S2/S3 block Certificate issuance. S4 failure sets `high-variance` flag. |
| **Triad gates as safety brakes** | [needs formalization] | `Physics_World_Model` | `packages/pwm_core/pwm_core/targeting/scoring.py`, `analysis/bottleneck.py` | Promote G1/G2/G3 from diagnostic annotations to certification gates. Hard G1/G2/G3 failure sets `safety-brake` risk flag; Certificate cannot reach Certified tier until flag is resolved or explicitly acknowledged. |
| **Benchmark trust tiers** | [new build] | `pwm_product` | `platform/pwm_platform/db/` + leaderboard routes | Add `trust_tier` and `risk_flags` columns to runs/submissions schema. Implement tier-promotion workflow. Render tier badges and risk-flag icons on leaderboard. Re-classify all 2,732 existing entries as Draft on migration. |
| **Golden reference bundles** | [new build] | `pwm_product` | `datasets/Benchmark/` + `platform/scripts/` | For each of the 12 Priority-1 modalities, produce one fully Certified RunBundle. These are the first rows to reach Certified tier and anchor all future submissions. |

**Exit criteria:**
- `certificate.json` is emitted for every completed run
- A run with incomplete spec (S1 fail) is blocked from producing a Certificate
- `trust_tier` column exists in the DB and is visible on the leaderboard
- 12 golden reference RunBundles exist and each has a Certificate at Certified tier

---

### P1 — First orbit ring

With the sun stable, add the first controlled orbits.

| Deliverable | State | Repo | Target package / file | Description |
|------------|-------|------|----------------------|-------------|
| **Docking artifact schemas** | [new build] | `Physics_World_Model` | `packages/pwm_core/pwm_core/cards/` | Pydantic/JSON Schema for SpecCard, MethodCard, DatasetCard, ClaimCard. P0 cards only (see Section 8). InstrumentCard and EventCard are P2. |
| **Primitive registries published** | [needs formalization] | `Physics_World_Model` | `packages/pwm_core/contrib/primitive_registry/` | After P-1 restructure, publish `general/v1`, `imaging/v1`, and `mappings/imaging_to_general/v1` as versioned, append-only YAML artifacts. CI enforces append-only rule. |
| **Interactive modality pages** | [needs formalization] | `pwm_product` | `platform/pwm_platform/routers/modalities.py` | Auto-generated from existing DomainProfiles. Add OperatorGraph DAG viewer, Triad sliders, algorithm table with trust-tier badges. Platform routes exist; badge rendering and DAG viewer are new. |
| **GitHub Action** | [new build] | `Physics_World_Model` | `.github/actions/pwm-benchmark/` | `pwm-benchmark` action in the public repo. Users can wire it against their own forks. Runs the 4-scenario protocol, already battle-tested. |
| **Controlled claim scaffolding** | [new build] | `pwm_product` | `platform/pwm_platform/services/arxiv_scaffolder/` | arXiv scanner (`eess.IV`, `physics.optics`, `cs.CV`) produces Draft ClaimCards. Review queue in the platform only — no auto-publish. |
| **CLI completion** | [new build] | `Physics_World_Model` | `packages/pwm_core/pwm_core/cli/` | Add `pwm synthesize` (data generation via existing forward models), `pwm ingest` (PHI strip + QC + DatasetCard emission), `pwm install` (plugin manager). Core CLI (`run`, `view`, `reproduce`, `doctor`) already exists. |

**Exit criteria:**
- One ClaimCard flows through scaffold → review queue → Draft tier on leaderboard
- One MethodCard can be submitted and benchmarked via `pwm run`
- `pwm ingest <dir>` emits a valid DatasetCard
- One primitive mapping artifact (`mappings/imaging_to_general/v1/`) is published and referenced by a running OperatorGraph

---

### P2 — Growth orbits

Expand the ecosystem with trust infrastructure in place.

| Deliverable | State | Repo | Target | Description |
|------------|-------|------|--------|-------------|
| **Plugin marketplace** | [new build] | `Physics_World_Model` + `pwm_product` | `packages/pwm_core/pwm_core/cli/install.py` + platform plugin registry | `pwm install` command (open), plugin signing, LIP-Arena-derived ratings (platform). |
| **Dataset federation** | [new build] | `Physics_World_Model` | `tools/dataset_federation/` | Federated registry indexing AAPM, fastMRI, BioImage Archive. PWM is the catalog, not the host. |
| **CT QC Copilot** | [needs formalization] | `Physics_World_Model` + `pwm_product` | `packages/pwm_core/contrib/domain_profiles/ct_qc/v1/` + platform QC routes | Early ingredients exist: `clinical_ct_thresholds.yaml` (7.8 KB) and `clinical_ct_mismatch.yaml` (19 KB) in `contrib/`. CT QC Copilot paper is in progress in `pwm/papers/ct_qc_copilot/`. Formalize as DomainProfile `ct_qc/v1` (open), add InstrumentCards, implement drift detection and compliance reports (product). First enterprise proof-of-concept after sun is hardened. |
| **Contributor economy** | [new build] | `pwm_product` | `platform/pwm_platform/db/` + UI routes | Roles, badges, contributor pages, maintainer rosters, challenge credits. DB schema + UI. |
| **Community & conference** | [new build] | — | — | Workshop proposals (MICCAI 2026, ISBI), monthly Grand Rounds, Weekly Digest auto-generation. P2+ only — see Section 7 note. |

**Exit criteria:**
- `pwm install <plugin>` installs and benchmarks a community solver
- One institution's CT scanner has an InstrumentCard and a running QC workflow
- Contributor pages are live with role + badge display
- One conference workshop proposal submitted

---

### P3 — Scale and expand

| Deliverable | State | Repo | Description |
|------------|-------|------|-------------|
| **Cloud IDE** | [new build] | `pwm_product` | Expand `pwm.platformai.org` into hosted compute with free academic tier. Permanent RunBundle URLs. |
| **Cross-domain expansion** | [new build] | `Physics_World_Model` | Acoustics, particle physics, remote sensing, materials, astronomy. Each adds a PrimitiveDialect + DomainProfile to the open repo. Imaging must be Certified-tier solid first. |
| **Autonomous science loops** | [new build] | `pwm` + `Physics_World_Model` | AI Scientist integration: hypothesis → experiment → evaluation → update. Research prototyped in `pwm`, deployed via open kernel. |
| **Hypothesis & transfer engines** | [new build] | `Physics_World_Model` | Triad-based hypothesis generation, cross-modality transfer suggestions. |

---

### First launch slice

A concrete path to the first public Dyson-swarm-credible release:

| Milestone | Repo | Description |
|-----------|------|-------------|
| **One Certified CASSI golden bundle** | `pwm_product` | CASSI is the most mature modality. Produce a fully Certified RunBundle with a visible Certificate on the leaderboard at `pwm.platformai.org`. This is the first public proof that the trust ratchet works. |
| **One Draft → Reproduced ClaimCard flow** | `pwm_product` | Take one arXiv CASSI paper, scaffold a Draft ClaimCard, confirm with author, independently reproduce, promote to Reproduced. Demonstrate the full tier lifecycle on the live platform. |
| **Trust-tier badge on leaderboard** | `pwm_product` | At least one leaderboard row shows a Certified or Reproduced badge in the UI. This is the public signal that the swarm is open. |
| **One emitted `certificate.json`** | `Physics_World_Model` | A real, machine-readable Certificate with all v1 fields, defined in the open repo. Linked from a RunBundle. Downloadable by anyone running the open kernel locally. |
| **`pwm-benchmark` GitHub Action demo** | `Physics_World_Model` | Published in the public repo. Any user can add it to their own repo to run the 4-scenario protocol on PRs. Opens the developer orbit. |
| **One DatasetCard from `pwm ingest`** | `Physics_World_Model` | Demonstrate that a user can bring their own imaging data and receive a machine-readable DatasetCard in one command from the open CLI. Opens the data orbit. |

---

## 12. Open-Core Model

PWM scales as a swarm only if the core protocol is open and the sustainability
model is clear. The three-repo structure (see Section 0) directly encodes the
open-core boundary.

### Open / public → `github.com/integritynoble/Physics_World_Model`

Everything in the public repo is open. Anyone can clone it and run the full
evaluation harness locally with identical trust guarantees to the hosted platform.

- CoreSpec schema and compiler
- OperatorGraph schema, compiler, and executor
- RunBundle schema and writer
- Certificate schema and issuer
- Primitive registries (`general/v1`, `imaging/v1`, mappings)
- Plugin interfaces and SDK
- CLI (`pwm run`, `pwm view`, `pwm ingest`, `pwm synthesize`, `pwm install`)
- Benchmark definitions and golden reference bundles
- Docking artifact schemas (all Cards)
- Judge kernel (S1-S4) and Triad gates
- Algorithm catalog (2,732 solvers, 172 modalities)
- All domain profiles and modality specs

### Private research → `github.com/integritynoble/pwm`

The private research repo holds the scientific work that drives the protocol
forward but is not part of the open kernel or the product:

- Academic papers (InverseNet, CT QC Copilot, Finite Primitive Theorem)
- Private experiments and ablations
- Architecture notes and internal design documents
- Unpublished prototype ideas before promotion to open kernel

**Promotion flow**: papers, once published, enter the public benchmark as
ClaimCards. Algorithms validated in private experiments are promoted into
`Physics_World_Model/algorithm_base/` before being cited or deployed.

### Product / paid → `github.com/integritynoble/pwm_product`

The product repo deploys the hosted platform at `pwm.platformai.org` and adds
commercial features on top of the open kernel. It contains a deployment copy of
`pwm_core` (see Section 0 sync risk note).

- **Managed cloud** (hosted compute, GPU scheduling, persistent storage) — `platform/`
- **Private workspaces** (team-scoped RunBundles, embargoed results) — `pwm_platform/auth/`
- **Hospital / enterprise connectors** (DICOM integration, PACS bridge, HL7/FHIR) — future
- **Premium benchmarking** (priority queue, large-scale sweeps, custom scenarios) — `pwm_platform/services/`
- **Institution dashboards** (multi-site QC overview, drift trending, fleet status) — future
- **Compliance / admin tooling** (audit export, role management, access control) — `pwm_platform/auth/`
- **Billing and subscription management** — `pwm_platform/routers/billing.py`
- **Support SLAs**

**Principle**: The protocol is free. The convenience is paid. Anyone can run
PWM locally from `Physics_World_Model` and get the same trust guarantees as the
hosted platform. The paid layer removes friction, adds scale, and provides
operational support — but must not redefine protocol semantics.

---

## 13. Regulatory Posture

PWM is not a regulatory approval mechanism. It is infrastructure that supports
evidence generation.

PWM can:

- **Support evidence generation** — produce structured, reproducible evaluation
  results that can be included in regulatory submissions.
- **Provide audit-grade traceability** — RunBundles and Certificates create a
  chain of evidence from input data to final metric, with cryptographic hashes.
- **Reduce validation-document burden** — standardized evaluation protocols
  mean less custom documentation per submission.
- **Align with compliance workflows** — the 4-scenario protocol and Triad
  analysis map onto concepts in FDA guidance for AI/ML-based SaMD and IEC
  quality management standards.

PWM does not:

- Claim to satisfy regulatory requirements on its own.
- Replace domain-specific validation required by authorities.
- Guarantee that a Certified result meets any particular regulatory standard.

The value proposition is: using PWM makes compliance evidence cheaper and more
rigorous to produce, not that PWM replaces the compliance process.

---

## 14. Anti-Patterns

Things PWM must not do:

1. **One parser per problem family.** Every new domain should reuse the CoreSpec
   compiler, not build a bespoke parser. If a new domain needs a parser, the
   CoreSpec is too narrow — fix the spec, not the toolchain.

2. **One flat primitive list mixing physics and computation.** The two-namespace
   model exists for a reason. Mixing "Propagate" and "Validate" in the same
   list conflates what the physics does with what the computer does. Keep them
   separate.

3. **Domain specs bypassing universal certification.** If a CT QC run can emit a
   Certificate without passing S1-S4, the trust kernel is broken. Every domain
   goes through the Judge. No exceptions.

4. **Auto-publishing noisy benchmark rows before trust tiers exist.** A
   leaderboard full of Draft-tier, unverified results destroys credibility
   faster than it builds it. Auto-scaffolding is fine. Auto-publishing is not.
   Trust tiers must be live before any public benchmark row appears.

5. **Designing for hypothetical future domains before the first domain is
   solid.** Computational imaging is the first domain. It must be Certified-tier
   solid before combustion, particle physics, or any other domain is added.
   Premature generalization is the enemy.

6. **Building convenience before trust.** A beautiful cloud IDE with broken
   reproducibility is worse than a CLI with perfect RunBundles. P0 is the sun.
   P3 is the chrome.

7. **Renaming the whole repo before the first trust ratchet works.** Compatibility
   aliases are acceptable. Massive terminology churn — renaming `ExperimentSpec`,
   restructuring `contrib/`, renaming CLI commands — before Certificate, hard
   gates, and trust tiers are live creates noise without value. P-1 alignment is
   a prerequisite pass, not a prolonged refactor. If the rename takes longer than
   the Certificate build, stop renaming and build the Certificate.

8. **Letting protocol logic diverge between `Physics_World_Model` and `pwm_product`.**
   `pwm_product` contains a deployment copy of `pwm_core`. If a kernel fix or
   new protocol object (Certificate schema, gate logic, Card schema) is applied
   only in `pwm_product` and never promoted back to `Physics_World_Model`, the
   sun splits into two incompatible versions. Product may add convenience and
   scale on top of the kernel. It must not silently redefine protocol semantics.
   If the public repo and the deployed repo disagree on what a Certificate means,
   the swarm has no canonical trust anchor.

---

## 15. Why This Is Feasible

The Dyson Swarm strategy is not a wishlist. The hardest parts are already built.

**The open kernel is already strong — `Physics_World_Model`:**
- **Physics engine and OperatorGraph IR** — 30+ primitives, typed DAG compiler,
  executor. Lives in `packages/pwm_core/pwm_core/graph/`. Used in every run today.
- **RunBundle** — immutable audit record with SHA-256 hashes, pip-freeze
  provenance, seeds, and full artifact directory. Lives in
  `packages/pwm_core/pwm_core/core/runbundle/`. Already emitted on every run.
- **4-scenario protocol** — fully implemented and battle-tested on CT, MRI,
  CASSI, CACTI, Ptychography, CryoEM, Ultrasound. Lives in `targeting/scenarios.py`.
- **Algorithm catalog** — 2,732 solvers across 172 modalities in `algorithm_base/`.
- **Registry** — 240 KB modality definitions, 97 KB compression tables, 73 KB
  mismatch distributions in `packages/pwm_core/contrib/`.

**The hosted platform is already real — `pwm_product`:**
- FastAPI platform deployed at `pwm.platformai.org` with routes, DB, auth,
  billing, and Docker Compose in `platform/pwm_platform/`.
- Deployment copy of `pwm_core` in `packages/pwm_core/` (see sync-risk note in
  Section 0 — must not diverge from the public kernel).

**Scientific framing is in progress — `pwm` (private research):**
- **InverseNet** — paper on calibration and mismatch correction (`papers/inversenet/`)
- **CT QC Copilot** — paper on the operations-flywheel CT QC system (`papers/ct_qc_copilot/`)
- **Finite Primitive Theorem** — foundational paper on the 11-primitive basis (`papers/finite_primitive_theorem/`)

These papers, once published, enter the public benchmark as ClaimCards and
anchor the scientific credibility of the trust kernel. New algorithms validated
here are promoted into `Physics_World_Model/algorithm_base/` before deployment.

**What remains is trust-ratchet engineering, not physics research:**
- Certificate object: one Pydantic model + emitter → `Physics_World_Model`, then synced to `pwm_product`
- Hard S1-S4 gates: data already captured — convert to verdicts → `Physics_World_Model`
- Trust tiers: one DB column, one promotion workflow, one badge renderer → `pwm_product`
- Card schemas: five Pydantic models (SpecCard, MethodCard, DatasetCard, ClaimCard, Certificate) → `Physics_World_Model`

The gap between "where we are" and "first Dyson-swarm-credible release" is
measured in engineering weeks, not research years. The open kernel is strong,
the platform is live, the papers are in progress. The solar collectors —
Certificate, trust tiers, and Card schemas — need fabrication.

---

## 16. The Dyson Swarm Principle

A Dyson Swarm does not try to own the star. It builds collectors that are
useful individually and devastating collectively.

PWM should not try to own all scientific objects. It should become the
**smallest shared protocol** that makes every scientific object easier to:

- **Express** — CoreSpec + DomainProfile + ProblemInstance
- **Run** — OperatorGraph → ComputePlan → execution
- **Compare** — standardized metrics, trust tiers, reproducible RunBundles
- **Trust** — Judge kernel, Certificates, independent reproduction
- **Reuse** — docking artifacts, plugin ecosystem, federated registries

The protocol is the sun. Everything else is orbit. Keep the sun small, stable,
and trust-centered. Let the orbits grow without bound.

PWM becomes a Dyson swarm only if the sun has one canonical home. In practice,
that means the protocol lives in `Physics_World_Model`, research matures in
`pwm`, and deployment scales through `pwm_product` without redefining the kernel.
The next step is not more breadth — it is turning existing evidence, provenance,
and evaluation machinery in `Physics_World_Model` into explicit trust objects
that outer orbits can dock to.

> *"PWM is not the answer to every scientific question.*
> *It is the smallest shared protocol that makes every answer cheaper to*
> *produce, easier to compare, and harder to fake."*
