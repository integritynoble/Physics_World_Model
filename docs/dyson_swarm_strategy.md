# PWM as a Dyson Swarm: Sun + Orbits Architecture

> A Dyson Swarm works only if the sun is stable. PWM becomes a swarm not by
> building many gravity wells, but by hardening one small, trust-centered kernel
> and letting everything else orbit it. The sun is the protocol. The orbits are
> the domains, datasets, methods, instruments, events, and communities that dock
> to it. If the sun drifts, every collector drifts. If the sun holds, collectors
> can be added, replaced, or removed without system-level damage.

---

## 1. What the Sun Is

The sun is not a modality, not a benchmark, not a product. The sun is the
**minimal shared protocol** that makes scientific objects expressible, executable,
comparable, trustable, and reusable.

### Sun objects (the canonical kernel)

| Object | Role |
|--------|------|
| **CoreSpec** | Universal six-tuple problem description: (object, forward model, measurement, noise model, prior, task). Domain-agnostic. |
| **Judge** | Universal trust kernel. Checks structural invariants S1-S4 on every run. Domain modules extend but never bypass it. |
| **OperatorGraph** | Intermediate representation. The compiled forward model as a typed DAG of primitives. |
| **RunBundle** | Immutable audit record. Inputs, outputs, logs, metrics, seeds, hashes — everything needed to reproduce or dispute a result. |
| **Certificate** | Machine-readable trust verdict. Issued by the Judge after a RunBundle passes all gates. Carries a trust tier. |
| **Registry** | Versioned catalog of primitives, profiles, datasets, methods, instruments. The namespace authority. |

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

**Design rule**: Every orbit element enters the system through a docking artifact
(see Section 8), passes through the Judge, and emits a RunBundle + Certificate.
No shortcut path exists.

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

| Gate | Check | Failure mode |
|------|-------|-------------|
| **S1: Spec completeness** | All six CoreSpec fields are bound and type-valid | Reject: incomplete specification |
| **S2: Reproducibility** | RunBundle contains sufficient information to reproduce the result (seeds, versions, hashes) | Reject: non-reproducible |
| **S3: Metric integrity** | Reported metrics match recomputed metrics from stored artifacts | Reject: metric tampering or drift |
| **S4: Budget compliance** | Compute, memory, and wall-clock stayed within declared bounds | Warn or reject: budget overrun |

### Domain Judge modules (additive, never subtractive)

| Domain | Additional gates |
|--------|-----------------|
| **Imaging** | Triad gates (G1: recoverability, G2: carrier budget, G3: operator mismatch), physics tier validation, 4-scenario consistency |
| **CT QC** | Threshold tables (noise, uniformity, CNR, MTF), artifact flags (ring, cupping, streak), drift detection against baseline |
| **Combustion** | Mass/energy conservation residuals, reaction rate plausibility, mesh convergence |
| **Spectroscopy** | Spectral resolution bounds, calibration lamp cross-check, baseline correction quality |

**Rule**: A domain module can add gates. It cannot weaken or skip S1-S4.
A Certificate is issued only when all kernel gates AND all active domain gates
pass. The Certificate records which gates were active, so trust is interpretable.

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
  certificate.json              # Trust verdict + tier
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
play. LIP-Arena already exists with 170 modalities and 2,755 algorithms. The
strategy is to harden it into the most trusted imaging benchmark in the field,
then use that credibility to expand.

- **Benchmark trust tiers** (every leaderboard row carries a tier):

| Tier | Meaning | Requirements |
|------|---------|-------------|
| **Draft** | Auto-scaffolded or self-reported. Not yet verified. | SpecCard + code link |
| **Author-confirmed** | Original authors have reviewed and accepted the PWM result. | Author sign-off on RunBundle |
| **Reproduced** | At least one independent party has reproduced the result within tolerance. | Independent RunBundle matching within declared epsilon |
| **Certified** | Passed full Judge (S1-S4 + domain gates). Golden-reference-quality. | Certificate issued |
| **Boundary-risk** | Result is near a trust boundary (e.g., metric variance overlaps with a rival method, or safety brake triggered). Event-horizon warning. | Automatic flag by Judge |

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

### Community orbit

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

| Card | Purpose | Key fields |
|------|---------|-----------|
| **SpecCard** | Declares a problem to be solved | CoreSpec subset, DomainProfile ref, task type |
| **MethodCard** | Declares a solver / algorithm | Name, version, code URI, primitive requirements, compute budget |
| **DatasetCard** | Declares a dataset | Modality, size, license, noise model, split structure, provenance |
| **InstrumentCard** | Declares a physical instrument | Manufacturer, model, primitive chain, calibration state |
| **ClaimCard** | Declares a result claim (from a paper, report, or experiment) | Source (DOI/arXiv), metric values, conditions, trust tier |
| **EventCard** | Declares a conference, workshop, challenge, or deadline | Date, venue, relevance, associated SpecCards |
| **RunBundle** | Immutable audit record of an executed run | (Sun object — see Section 1) |
| **Certificate** | Trust verdict from the Judge | (Sun object — see Section 1) |

**Lifecycle**: Card → compile → run → RunBundle → Judge → Certificate.
A card that never compiles and runs stays at Draft tier. A card that runs and
passes the Judge earns a Certificate. This is the trust ratchet.

---

## 9. Contributor Economy

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

### P0 — Harden the sun

Nothing should orbit an unstable sun. P0 is entirely about kernel integrity.

| Deliverable | Description |
|------------|-------------|
| **Schema hardening** | Finalize CoreSpec, DomainProfile, ProblemInstance as JSON Schema / Pydantic models. Semver-lock `spec.core`. |
| **Sun object model** | Implement typed objects: CoreSpec, OperatorGraph, ComputePlan, RunBundle, Certificate, TriadReport. Markdown rendering is derived, not authoritative. |
| **Judge kernel** | Implement S1-S4 universal gates. Wire them into every code path. No run completes without Judge sign-off. |
| **Canonical IR** | Harden OperatorGraph as domain-neutral DAG with `(registry, version, name)` primitive references. |
| **RunBundle / Certificate** | Immutable, SHA-256 hashed, self-contained. Every run emits both. |
| **Benchmark trust tiers** | Implement Draft / Author-confirmed / Reproduced / Certified / Boundary-risk tiers. All existing 2,755 algorithm entries are re-classified. |
| **Golden reference bundles** | For each Priority-1 modality (12), produce one fully Certified RunBundle as the trust anchor. |

### P1 — First orbit ring

With the sun stable, add the first controlled orbits.

| Deliverable | Description |
|------------|-------------|
| **Primitive registries** | Publish `general/v1` (12 primitives), `imaging/v1` (11 primitives), `mappings/imaging_to_general/v1`. |
| **Interactive modality pages** | Auto-generated from DomainProfiles. OperatorGraph viewer, Triad sliders, algorithm comparison. |
| **GitHub Action** | `pwm-benchmark` action. 4-scenario protocol on PR. |
| **Controlled claim scaffolding** | arXiv scanner produces Draft ClaimCards. Review queue, not auto-publish. |
| **Docking artifact schemas** | Publish SpecCard, MethodCard, DatasetCard, InstrumentCard, ClaimCard, EventCard schemas. |

### P2 — Growth orbits

Expand the ecosystem with trust infrastructure in place.

| Deliverable | Description |
|------------|-------------|
| **Plugin marketplace** | Formalized `contrib/` with `pwm install`, LIP-Arena-derived ratings. |
| **Dataset federation** | Federated registry indexing AAPM, fastMRI, BioImage Archive, etc. |
| **Community & conference** | Workshop proposals (MICCAI 2026, ISBI), monthly Grand Rounds, Weekly Digest. |
| **CT QC Copilot** | First operations-flywheel vertical. DomainProfile `ct_qc/v1`, InstrumentCards for major CT scanners, drift detection, compliance reports. |
| **Contributor economy** | Roles, badges, contributor pages, maintainer rosters, challenge credits. |

### P3 — Scale and expand

| Deliverable | Description |
|------------|-------------|
| **Cloud IDE** | Expand `pwm.platformai.org` into hosted compute with free academic tier. Permanent RunBundle URLs. |
| **Cross-domain expansion** | Acoustics, particle physics, remote sensing, materials, astronomy. Each adds a PrimitiveDialect + DomainProfile. |
| **Autonomous science loops** | AI Scientist integration: hypothesis → experiment → evaluation → update. |
| **Hypothesis & transfer engines** | Triad-based hypothesis generation, cross-modality transfer suggestions. |

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

---

## 15. The Dyson Swarm Principle (Revised)

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

> *"PWM is not the answer to every scientific question.*
> *It is the smallest shared protocol that makes every answer cheaper to*
> *produce, easier to compare, and harder to fake."*
