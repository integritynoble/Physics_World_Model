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

### Current state summary

| Component | Status | Location |
|-----------|--------|----------|
| OperatorGraph IR (30+ primitives, typed DAG, compiler, executor) | **~95% built** | `packages/pwm_core/pwm_core/graph/` |
| RunBundle (SHA-256 hashes, provenance, manifest, artifact storage) | **~95% built** | `packages/pwm_core/pwm_core/core/runbundle/` |
| Algorithm catalog (2,732 solvers, 172 modalities) | **100% built** | `algorithm_base/` |
| 4-scenario protocol (Ideal / Assumed / Corrected / Oracle) | **100% built** | `packages/pwm_core/pwm_core/targeting/scenarios.py` |
| Registry (primitives, modalities, datasets, solver routing) | **~100% built** | `packages/pwm_core/contrib/` |
| CoreSpec (`ExperimentSpec v0.2.1`, 172 modality domain profiles) | **~90% built** | `spec/` |
| Web platform (routes, DB, auth, billing, Docker) | **~90% built** | `platform/pwm_platform/` |
| Judge S1-S4 data (validation reports, provenance, hashes, BudgetGuard) | **data captured, gates not wired** | `targeting/harness.py`, `budget.py` |
| Triad gates G1-G3 (bottleneck classification) | **scoring only, not safety brakes** | `targeting/scoring.py`, `analysis/bottleneck.py` |
| Certificate object | **not yet built** | — |
| Benchmark trust tiers | **documented, not in DB or UI** | — |
| Docking artifact schemas (Cards) | **not yet built** | — |
| CLI `pwm synthesize / ingest / install` | **not yet built** | `packages/pwm_core/pwm_core/cli/` |

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

| Deliverable | Current state | Action | Target |
|------------|--------------|--------|--------|
| **CoreSpec alias** | `ExperimentSpec v0.2.1` | Introduce `CoreSpec` as compatibility-preserving alias; begin splitting into CoreSpec / DomainProfile / ProblemInstance layers | `packages/pwm_core/pwm_core/spec/core.py` |
| **Primitive registry layout** | `contrib/primitives.yaml` + `graph/primitives.py` | Reorganize into `primitive_registry/general/v1/`, `imaging/v1/`, `mappings/imaging_to_general/v1/` | `packages/pwm_core/contrib/primitive_registry/` |
| **OperatorGraph primitive refs** | Internal string names | Switch to `(registry, version, name)` triples | `packages/pwm_core/pwm_core/graph/ir_types.py` |

**Exit criteria:**
- `from pwm_core.spec import CoreSpec` works; existing `ExperimentSpec` callers unbroken
- Primitive registry files exist at target paths; old paths symlinked or aliased
- CI passes with no regressions

---

### P0 — Complete the sun *(5 genuine gaps)*

The OperatorGraph IR, RunBundle, 4-scenario protocol, algorithm catalog (2,732
solvers, 172 modalities), and platform are already production-ready. P0 addresses
only the five things that are genuinely missing from the trust kernel.

| Deliverable | State | Target package / file | Description |
|------------|-------|----------------------|-------------|
| **Certificate object** | [new build] | `pwm_core/core/runbundle/certificate.py` | Pydantic model + `certificate.json` emitter. See Section 4 for required fields. Wire into `harness.py` so every completed run emits one. |
| **S1-S4 as hard gates** | [needs formalization] | `targeting/harness.py`, `budget.py`, `provenance.py` | Convert existing data captures (validation report, provenance, SHA-256 hashes, BudgetGuard) into explicit pass/fail/warn verdicts. Failed S1/S2/S3 block Certificate issuance. S4 failure sets `high-variance` flag. |
| **Triad gates as safety brakes** | [needs formalization] | `targeting/scoring.py`, `analysis/bottleneck.py` | Promote G1/G2/G3 from diagnostic annotations to certification gates. Hard G1/G2/G3 failure sets `safety-brake` risk flag; Certificate cannot reach Certified tier until flag is resolved or explicitly acknowledged. |
| **Benchmark trust tiers** | [new build] | `platform/pwm_platform/db/` + leaderboard routes | Add `trust_tier` and `risk_flags` columns to runs/submissions schema. Implement tier-promotion workflow. Render tier badges and risk-flag icons on leaderboard. Re-classify all 2,732 existing entries as Draft on migration. |
| **Golden reference bundles** | [new build] | `benchmark_results/golden/` | For each of the 12 Priority-1 modalities, produce one fully Certified RunBundle. These are the first rows to reach Certified tier and anchor all future submissions. |

**Exit criteria:**
- `certificate.json` is emitted for every completed run
- A run with incomplete spec (S1 fail) is blocked from producing a Certificate
- `trust_tier` column exists in the DB and is visible on the leaderboard
- 12 golden reference RunBundles exist and each has a Certificate at Certified tier

---

### P1 — First orbit ring

With the sun stable, add the first controlled orbits.

| Deliverable | State | Target package / file | Description |
|------------|-------|----------------------|-------------|
| **Docking artifact schemas** | [new build] | `pwm_core/cards/` | Pydantic/JSON Schema for SpecCard, MethodCard, DatasetCard, ClaimCard. P0 cards only (see Section 8). InstrumentCard and EventCard are P2. |
| **Primitive registries published** | [needs formalization] | `contrib/primitive_registry/` | After P-1 restructure, publish `general/v1`, `imaging/v1`, and `mappings/imaging_to_general/v1` as versioned, append-only YAML artifacts. CI enforces append-only rule. |
| **Interactive modality pages** | [needs formalization] | `platform/pwm_platform/routers/modalities.py` | Auto-generated from existing DomainProfiles. Add OperatorGraph DAG viewer, Triad sliders, algorithm table with trust-tier badges. Platform routes exist; badge rendering and DAG viewer are new. |
| **GitHub Action** | [new build] | `.github/actions/pwm-benchmark/` | `pwm-benchmark` action runs the 4-scenario protocol on PRs. 4-scenario protocol is already battle-tested. |
| **Controlled claim scaffolding** | [new build] | `tools/arxiv_scaffolder/` | arXiv scanner (`eess.IV`, `physics.optics`, `cs.CV`) produces Draft ClaimCards. Review queue only — no auto-publish. |
| **CLI completion** | [new build] | `pwm_core/cli/` | Add `pwm synthesize` (data generation via existing forward models), `pwm ingest` (PHI strip + QC + DatasetCard emission), `pwm install` (plugin manager). Core CLI (`run`, `view`, `reproduce`, `doctor`) already exists. |

**Exit criteria:**
- One ClaimCard flows through scaffold → review queue → Draft tier on leaderboard
- One MethodCard can be submitted and benchmarked via `pwm run`
- `pwm ingest <dir>` emits a valid DatasetCard
- One primitive mapping artifact (`mappings/imaging_to_general/v1/`) is published and referenced by a running OperatorGraph

---

### P2 — Growth orbits

Expand the ecosystem with trust infrastructure in place.

| Deliverable | State | Target | Description |
|------------|-------|--------|-------------|
| **Plugin marketplace** | [new build] | `contrib/` + `pwm_core/cli/install.py` | `pwm install` command, plugin signing, LIP-Arena-derived ratings. |
| **Dataset federation** | [new build] | `tools/dataset_federation/` | Federated registry indexing AAPM, fastMRI, BioImage Archive. PWM is the catalog, not the host. |
| **CT QC Copilot** | [needs formalization] | `domain_profiles/ct_qc/v1/` | Early ingredients exist: `clinical_ct_thresholds.yaml` (7.8 KB) and `clinical_ct_mismatch.yaml` (19 KB) in `contrib/`. This is the first operations-flywheel vertical — not yet a full orbit. Formalize as DomainProfile `ct_qc/v1`, add InstrumentCards for major CT scanners, implement drift detection, generate compliance reports. Treat this as the first enterprise proof-of-concept after the sun is hardened, not a launch blocker. |
| **Contributor economy** | [new build] | `platform/pwm_platform/` | Roles, badges, contributor pages, maintainer rosters, challenge credits. DB schema + UI. |
| **Community & conference** | [new build] | — | Workshop proposals (MICCAI 2026, ISBI), monthly Grand Rounds, Weekly Digest auto-generation. P2+ only — see Section 7 note. |

**Exit criteria:**
- `pwm install <plugin>` installs and benchmarks a community solver
- One institution's CT scanner has an InstrumentCard and a running QC workflow
- Contributor pages are live with role + badge display
- One conference workshop proposal submitted

---

### P3 — Scale and expand

| Deliverable | State | Description |
|------------|-------|-------------|
| **Cloud IDE** | [new build] | Expand `pwm.platformai.org` into hosted compute with free academic tier. Permanent RunBundle URLs. |
| **Cross-domain expansion** | [new build] | Acoustics, particle physics, remote sensing, materials, astronomy. Each adds a PrimitiveDialect + DomainProfile. Imaging must be Certified-tier solid first. |
| **Autonomous science loops** | [new build] | AI Scientist integration: hypothesis → experiment → evaluation → update. |
| **Hypothesis & transfer engines** | [new build] | Triad-based hypothesis generation, cross-modality transfer suggestions. |

---

### First launch slice

A concrete path to the first public Dyson-swarm-credible release:

| Milestone | Description |
|-----------|-------------|
| **One Certified CASSI golden bundle** | CASSI is the most mature modality. Produce a fully Certified RunBundle with a visible Certificate on the leaderboard. This is the first public proof that the trust ratchet works. |
| **One Draft → Reproduced ClaimCard flow** | Take one arXiv CASSI paper, scaffold a Draft ClaimCard, confirm with author, independently reproduce, promote to Reproduced. Demonstrate the full tier lifecycle. |
| **Trust-tier badge on leaderboard** | At least one leaderboard row shows a Certified or Reproduced badge in the UI. This is the public signal that the swarm is open. |
| **One emitted `certificate.json`** | A real, machine-readable Certificate with all v1 fields, linked from a RunBundle. Downloadable by anyone. |
| **`pwm-benchmark` GitHub Action demo** | A public repo demonstrates the action running the 4-scenario protocol on a PR. Opens the developer orbit. |
| **One DatasetCard from `pwm ingest`** | Demonstrate that a user can bring their own imaging data and receive a machine-readable DatasetCard in one command. Opens the data orbit. |

---

## 12. Open-Core Model

PWM scales as a swarm only if the core protocol is open and the sustainability
model is clear.

### Open (public, permissive license)

- `spec.core.md` / CoreSpec schema
- OperatorGraph schema and compiler
- RunBundle schema
- Certificate schema
- Primitive registries (`general/v1`, `imaging/v1`, mappings)
- Plugin interfaces and SDK
- CLI (`pwm run`, `pwm view`, `pwm ingest`, `pwm synthesize`)
- Benchmark definitions and golden reference bundles
- Docking artifact schemas (all Cards)
- Judge kernel (S1-S4)

### Private / paid (sustainability layer)

- Managed cloud (hosted compute, GPU scheduling, persistent storage)
- Private workspaces (team-scoped RunBundles, embargoed results)
- Hospital / enterprise connectors (DICOM integration, PACS bridge, HL7/FHIR)
- Premium benchmarking (priority queue, large-scale sweeps, custom scenarios)
- Institution dashboards (multi-site QC overview, drift trending, fleet status)
- Compliance / admin tooling (audit export, role management, access control)
- Support SLAs

**Principle**: The protocol is free. The convenience is paid. Anyone can run
PWM locally and get the same trust guarantees. The paid layer removes friction,
adds scale, and provides operational support.

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

---

## 15. Why This Is Feasible

The Dyson Swarm strategy is not a wishlist. The hardest parts are already built.

**Already in production:**
- **Physics engine and OperatorGraph IR** — 30+ primitives, typed DAG compiler,
  executor. Used in every benchmark run today.
- **RunBundle** — immutable audit record with SHA-256 hashes, pip-freeze
  provenance, seeds, and a full artifact directory. Already emitted on every run.
- **4-scenario protocol** — fully implemented and battle-tested on CT, MRI,
  CASSI, CACTI, Ptychography, CryoEM, Ultrasound, and more.
- **Algorithm catalog** — 2,732 solvers across 172 modalities, importable and
  callable today.
- **Web platform** — deployed at `pwm.platformai.org` with routes, DB, auth,
  billing, and Docker Compose.
- **Registry** — 240 KB modality definitions, 97 KB compression tables, 73 KB
  mismatch distributions, solver routing for all modalities.

**What remains is trust-ratchet engineering, not physics research:**
- Certificate object: one new Pydantic model and one emitter wired into the
  existing harness
- Hard S1-S4 gates: data already captured — convert to verdicts
- Trust tiers: one DB column, one promotion workflow, one badge renderer
- Card schemas: five Pydantic models for SpecCard, MethodCard, DatasetCard,
  ClaimCard, Certificate

The gap between "where we are" and "first Dyson-swarm-credible release" is
measured in engineering weeks, not research years. The sun is mostly built. The
solar collectors need fabrication.

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

PWM is feasible as a Dyson swarm because the sun is already mostly built. The
next step is not more breadth — it is turning existing evidence, provenance, and
evaluation machinery into explicit trust objects that outer orbits can dock to.

> *"PWM is not the answer to every scientific question.*
> *It is the smallest shared protocol that makes every answer cheaper to*
> *produce, easier to compare, and harder to fake."*
