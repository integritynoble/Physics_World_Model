# PWM Lock-In Strategy: Building the Rail for Computational Imaging

> Reference: [https://solveeverything.org/](https://solveeverything.org/)
>
> Date: 2026-02-18
>
> Status: Strategic Framework

---

## Executive Summary

PWM (Physics World Model) is not a moonshot listed on solveeverything.org. It is something potentially more valuable: a **rail** — the institutional infrastructure that multiple moonshots ride on. Every moonshot in biology, materials science, manufacturing, and planetary monitoring requires calibrated imaging pipelines. PWM provides them.

The solveeverything.org framework identifies five institutional primitives that underpin the abundance economy. PWM maps directly to three of them. The 18-month "Regulatory Foundry Window" described in the framework is the critical period to establish PWM as the default evaluation infrastructure for computational imaging — achieving lock-in before the standards harden.

---

## Part 1: Rails vs Trains — Why PWM's Position Is Durable

### The solveeverything.org Thesis

> "The durable, compounding value in this new economy will not be found in owning any single AI model. Models are destined to be commoditized; they are the 'trains' that will eventually all look the same. The real value lies in owning the 'rails': the targeting platforms, the audit infrastructure, the data trusts, the action networks, and the compute escrow services."

### PWM Is Not a Train

Reconstruction methods (MST-L, HDNet, GAP-TV, future architectures) are **trains**. They will be commoditized. Today's state-of-the-art (34.81 dB) will be tomorrow's baseline. No single reconstruction method provides durable value.

### PWM Is a Rail

PWM provides the **infrastructure** that all reconstruction methods are evaluated on, calibrated through, and deployed via. Specifically:

| Institutional Primitive | PWM Implementation | Lock-in Mechanism |
|------------------------|-------------------|-------------------|
| **Targeting Authority** | 4-scenario evaluation protocol (Ideal, Assumed, Corrected, Oracle), PSNR/SSIM/SAM metrics, adversarial mismatch injection, mask-sensitivity spectrum | First standardized imaging benchmark spanning 64 modalities. Whoever defines the metrics defines the economy. |
| **Action Network** | Forward models + reconstruction solvers as shared digital laboratory. PhysicsOperator protocol spanning X-ray, MRI, optical, radar, electron, particle imaging. | The API surface that connects digital inference to physical imaging systems. Every new method must implement PhysicsOperator to be evaluated. |
| **Data Trust** | YAML registries (model_id + parameters, no eval), JSON result files, Replication Packs (scripts + data + configs), StrictBaseModel with extra="forbid" | Auditable, reproducible, mechanically validated. No freeform strings. Every result comes with a complete replication pack. |

### Why Moonshots Need This Rail

Every moonshot on solveeverything.org that involves sensing the physical world needs calibrated imaging:

| Moonshot Domain | Imaging Need | PWM Modalities |
|----------------|-------------|----------------|
| **Biology & Medicine** (Domain 4) | MRI for brain mapping, CT for diagnostics, ultrasound for monitoring, OCT for ophthalmology | fMRI, DW-MRI, MRS, CBCT, Doppler ultrasound, elastography, fundus, OCT-A, endoscopy (18 medical modalities) |
| **Chemistry & Materials** (Domain 3) | Electron microscopy for atomic structure, spectroscopy for composition | Electron diffraction, EBSD, EELS, electron holography, two-photon, STED, PALM/STORM (12 characterization modalities) |
| **Planetary Scale** (Domain 6) | SAR for earth observation, LiDAR for terrain, sonar for ocean | SAR, LiDAR, structured light, sonar, ToF camera (7 remote sensing modalities) |
| **Manufacturing** (Domain 5) | Inspection, quality control, digital twins | Neutron tomography, proton radiography, muon tomography, X-ray CT (8 inspection modalities) |
| **Physics** (Domain 2) | Hyperspectral imaging, coded aperture systems, compressed sensing | CASSI, SPC, CACTI (3 compressive modalities + 64 total) |

**When a new moonshot launches in any of these domains, it must ride on a calibrated imaging pipeline. That pipeline is PWM.**

---

## Part 2: The AlphaFold Parallel — Industrial Intelligence Stack

### How AlphaFold Achieved Domain Collapse

AlphaFold collapsed structural biology because it had all four layers of the Industrial Intelligence Stack:

1. **Purpose**: Predict protein 3D structure from amino acid sequence
2. **Task Taxonomy**: Sequence → (X, Y, Z) coordinates — a precise mathematical problem
3. **Observability**: Protein Data Bank — 200,000+ solved structures as training data
4. **Targeting System**: CASP — biennial, blinded, adversarial competition

### PWM's Industrial Intelligence Stack

| Layer | AlphaFold (Biology) | PWM (Imaging) — Current | PWM — Needed for Lock-In |
|-------|-------------------|------------------------|--------------------------|
| **Purpose** | Predict protein structure | Calibrate & reconstruct any imaging modality | Already clear and formalized |
| **Task Taxonomy** | Sequence → coordinates | ExperimentSpec → Validated Reconstruction (Pydantic schema, registry IDs) | Already formalized (v0.2.1) |
| **Observability** | Protein Data Bank (200K+ structures) | KAIST benchmark (10 scenes, 256x256x28), TSA masks, 6 YAML registries | **Critical Gap**: Need a "Physics Imaging Data Bank" with 1000+ scenes across modalities |
| **Targeting System** | CASP competition (biennial, blinded, adversarial) | 4-scenario protocol (internal, 5 methods, 10 scenes) | **Critical Gap**: Need a public, adversarial, rolling competition (CISP) |

### The Two Critical Gaps

#### Gap 1: The Shared Corpus — "Physics Imaging Data Bank"

PWM currently validates on 10 KAIST scenes with 1 mask type. For domain collapse, we need:

- **Scale**: 1000+ scenes across visible, infrared, X-ray, MRI, ultrasound
- **Diversity**: Multiple mask types (binary, grayscale, learned, random), multiple noise models, multiple mismatch profiles
- **Community-contributed**: Not just one lab's data — a shared resource like the Protein Data Bank
- **Standardized format**: NPZ/HDF5 with metadata schema, provenance fields, license terms
- **Privacy-preserving**: Synthetic scenes for medical imaging (no patient data), federated access for sensitive datasets

This corpus becomes the **observability layer** that makes PWM the default infrastructure. Whoever hosts this corpus controls the rail.

#### Gap 2: The Public Competition — "CISP"

CASP (Critical Assessment of Structure Prediction) made AlphaFold possible. PWM needs **CISP** — Critical Assessment of Spectral/Spatial Imaging Prediction.

CISP design principles (from solveeverything.org framework):

| Principle | CISP Implementation |
|-----------|-------------------|
| **Outcome-Grounded** | Not just PSNR — measure diagnostic accuracy (medical), defect detection rate (industrial), classification accuracy (remote sensing) |
| **Prospective & Blinded** | Test on scenes the model has never seen. Cryptographically committed holdouts. Rolling submissions, not annual. |
| **Adversarial & Anti-Gaming** | Red-team budget to inject adversarial mismatch, noise, and distribution shifts. Multi-objective Pareto scoring (accuracy + calibration + speed + fairness). |
| **Auditable & Equity-Constrained** | DR-AIS (Decision Records) for all submissions. Fairness bands: method must work across all scene types, not just "easy" ones. Replication Packs mandatory. |
| **Continuous** | 24/7 automated scoring. Weekly leaderboard updates. Rolling test set rotation managed by independent stewards. |

CISP Tracks:
1. **Spectral Imaging**: CASSI reconstruction under mismatch (5-parameter calibration)
2. **Temporal Imaging**: CACTI video reconstruction under motion blur
3. **Compressed Sensing**: SPC reconstruction under measurement noise
4. **Medical Imaging**: MRI/CT reconstruction under acquisition artifacts
5. **Remote Sensing**: SAR/LiDAR reconstruction under atmospheric distortion
6. **Cross-Modal**: Transfer performance across modality families

---

## Part 3: The Abundance Flywheel — PWM Edition

### The Generic Flywheel (solveeverything.org)

```
Commitment → Focus → Collapse → Surplus → Reinvestment → (repeat)
```

### PWM's Flywheel

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. COMMITMENT                                              │
│     Lab pre-commits GPU hours for imaging calibration       │
│     Compute Escrow: funds locked against imaging targets    │
│                          │                                  │
│                          ▼                                  │
│  2. FOCUS                                                   │
│     Research targets specific Time-to-Reconstruction        │
│     All R&D measured on CISP leaderboard                    │
│     Capital flows to methods that clear targets             │
│                          │                                  │
│                          ▼                                  │
│  3. COLLAPSE                                                │
│     Calibration becomes automated (CASSI paper proves       │
│     this: self-supervised, 5 min/scene, +3 dB recovery)    │
│     Imaging goes from PhD-craft to compute-query            │
│                          │                                  │
│                          ▼                                  │
│  4. SURPLUS                                                 │
│     Imaging cost drops from PhD-years to GPU-minutes        │
│     New business models: pay-per-reconstruction,            │
│     calibration-as-a-service, imaging SLAs                  │
│                          │                                  │
│                          ▼                                  │
│  5. REINVESTMENT                                            │
│     Surplus funds next modality (MRI, CT, SAR...)           │
│     Each cycle adds modalities to the platform              │
│     64 modalities → 100 → 200 → universal coverage         │
│                          │                                  │
│                          ▼                                  │
│     Back to Step 1, but now spanning more modalities        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Flywheel Metrics

| Cycle Stage | Metric | Current Value | Lock-In Target |
|------------|--------|---------------|----------------|
| Commitment | GPU-hours escrowed against imaging targets | Internal only | Public escrow, 10+ labs |
| Focus | Methods submitted to CISP per quarter | 5 (internal) | 50+ (community) |
| Collapse | Time-to-Reconstruction (seconds) | 484 ± 45 sec | < 60 sec |
| Surplus | Cost per validated reconstruction | ~$0.50 (GPU time) | < $0.01 |
| Reinvestment | Modalities covered per cycle | 64 | +10 per quarter |

### Safety as Mechanical Component

The flywheel includes **programmatic safety** (not afterthought safety):

- **Automatic downshift**: If calibration accuracy regresses on any subgroup (scene type, noise level, modality family), the system throttles automatically
- **Multi-compiler rule**: Safety-critical imaging (medical, nuclear) requires two independent reconstruction pipelines to agree — PWM's "Two-Stack Rule"
- **Equity constraint**: Methods must clear PSNR floors across ALL scene types, not just aggregate. A method that works on "easy" scenes but fails on "hard" scenes is rejected
- **Kill switch**: If mismatch-corrected reconstruction (Sc.III) ever performs WORSE than uncorrected (Sc.II), calibration is halted and flagged for human review

---

## Part 4: The Lock-In Path — 18-Month Roadmap

The solveeverything.org framework describes an 18-month "Regulatory Foundry Window" where standards harden. PWM must establish its position within this window.

### Phase 1: Become the Default Evaluation Infrastructure (Months 1-6)

**Goal**: Make PWM's 4-scenario protocol the standard way to evaluate computational imaging methods.

| Action | Deliverable | Lock-In Effect |
|--------|-----------|----------------|
| Open-source the 4-scenario evaluation protocol as standalone package | `pip install pwm-eval` | First mover on evaluation standard (QWERTY effect) |
| Publish Replication Packs for CASSI, SPC, CACTI | 3 complete packs with code + data + results | Establishes the "what good looks like" reference |
| Get 3-5 external labs to validate their methods on PWM protocol | Published papers citing PWM evaluation | Network effect: once 5 labs use it, the 6th has no choice |
| Define the mask-sensitivity spectrum as a standard metric | Adopted in 2+ conference papers | PWM vocabulary becomes the field's vocabulary |
| Publish DR-AIS templates for imaging pipelines | Downloadable templates | Compliance infrastructure that others build on |

**The QWERTY Moment**: The first evaluation protocol that gets adopted by 5+ labs becomes the permanent standard. Every subsequent method will be measured on PWM's terms. This is the lock-in.

### Phase 2: Launch CISP — The Public Competition (Months 7-12)

**Goal**: Create the "CASP for imaging" — a public, adversarial, rolling competition.

| Action | Deliverable | Lock-In Effect |
|--------|-----------|----------------|
| Launch CISP website with rolling submission system | cisp.pwm.org (or similar) | Gravity well: all imaging R&D focuses here |
| Establish independent steward board (3-5 neutral parties) | Published governance charter | Trust + anti-gaming credibility |
| Create blinded test sets managed by stewards | Cryptographically committed holdouts | Prospective testing prevents memorization |
| Define multi-track competition (spectral, temporal, medical, remote) | 4+ tracks with distinct metrics | Broad coverage prevents single-method dominance |
| Implement automated scoring pipeline (24/7, rolling) | API-based submission + instant scoring | Continuous operation (not annual pageant) |
| Fund red-team endowment | Public bounties for exploit discovery | Adversarial hardening of the ecosystem |

**The CASP Moment**: Once CISP has 20+ submissions and produces a leaderboard that conferences cite, it becomes self-sustaining. Methods not evaluated on CISP lack credibility.

### Phase 3: Become the Action Network (Months 13-18)

**Goal**: PWM's PhysicsOperator protocol becomes the API that connects digital inference to physical imaging systems.

| Action | Deliverable | Lock-In Effect |
|--------|-----------|----------------|
| Partner with 2-3 robotic labs for closed-loop imaging | API integration: PWM spec → robot measurement → PWM reconstruction → calibration → repeat | Physical action surface for digital intelligence |
| Implement Compute Escrow for imaging targets | Smart contract: GPU hours released on target clear | Financial primitive tied to PWM metrics |
| Launch calibration-as-a-service API | `POST /calibrate` with measurement + mask → calibrated parameters | Revenue model + API dependency |
| Establish outcome-based imaging contracts | "Pay per validated reconstruction" model | Procurement innovation: outcomes not effort |
| Expand Physics Imaging Data Bank to 500+ scenes | Community-contributed, standardized format | Observability layer that makes PWM indispensable |

**The Platform Moment**: Once physical labs route measurements through PWM's API, and funding flows through PWM's escrow, the lock-in is structural. Switching costs become prohibitive.

---

## Part 5: PWM's Position in the solveeverything.org Ecosystem

### The Ecosystem Map

```
solveeverything.org Moonshots
            │
            ├── Biology & Medicine ──────┐
            │   (cure disease,           │
            │    extend healthspan)       │
            │                            │
            ├── Chemistry & Materials ───┤
            │   (inverse design,         │
            │    new batteries)          │
            │                            ├──→ All require IMAGING
            ├── Manufacturing ───────────┤         │
            │   (D2P24, zero-defect)     │         │
            │                            │         ▼
            ├── Planetary Scale ─────────┤    ┌─────────┐
            │   (climate, energy,        │    │   PWM   │
            │    food, water)            │    │  (Rail) │
            │                            │    └────┬────┘
            └── Physics & Cosmology ─────┘         │
                (simulation, prediction)           │
                                                   ▼
                                          ┌────────────────┐
                                          │ PhysicsOperator │
                                          │    Protocol     │
                                          │  64 modalities  │
                                          │  89 templates   │
                                          │  4-scenario     │
                                          │  evaluation     │
                                          └────────────────┘
```

### PWM's Value Proposition by Stakeholder

| Stakeholder | What PWM Provides | solveeverything.org Role |
|------------|-------------------|------------------------|
| **Researchers** | Standardized evaluation (4-scenario), reproducible baselines, Replication Packs | Targeting Authority |
| **Labs** | Calibration-as-a-service, forward model library, reconstruction solver benchmarks | Action Network |
| **Funders** | Compute Escrow against imaging targets, measurable ROI (dB per GPU-dollar) | Compute Escrow |
| **Policymakers** | DR-AIS templates, compliance streams, equity constraints | Audit Infrastructure |
| **Industry** | Pay-per-reconstruction APIs, imaging SLAs, digital twin infrastructure | Outcome Procurement |

### What PWM Does NOT Do (Staying on the Rail)

PWM must resist the temptation to become a train:

- **PWM does NOT develop new reconstruction methods** — it evaluates them
- **PWM does NOT own imaging data** — it provides the trust framework for sharing
- **PWM does NOT compete with specific labs** — it provides the infrastructure all labs use
- **PWM does NOT pick winners** — the targeting system picks winners based on outcomes

> "Define the 'what' (the outcome) and let the market invent the 'how.'"

---

## Part 6: Failure Modes and Countermeasures

### Failure Mode 1: Spec Capture

**Risk**: PWM's metrics stop reflecting real-world imaging quality. Labs optimize for PSNR but produce clinically useless reconstructions.

**Countermeasure**: Multi-objective Pareto scoring. PSNR alone is not enough — must also clear diagnostic accuracy (medical), defect detection rate (industrial), and classification accuracy (remote sensing). Rotate independent stewards annually.

### Failure Mode 2: Monoculture

**Risk**: Everyone uses the same reconstruction method (e.g., MST-L) because it tops the CISP leaderboard. A bug in MST-L crashes all imaging pipelines.

**Countermeasure**: Multi-compiler rule for safety-critical domains. PWM's Two-Stack Rule: no reconstruction goes live until two independent methods agree. The mask-sensitivity spectrum already characterizes this risk — HDNet and GAP-TV provide fundamentally different failure modes than MST-L.

### Failure Mode 3: Data Leakage

**Risk**: Reconstruction methods memorize KAIST scenes and perform poorly on novel data.

**Countermeasure**: Rolling, cryptographically committed holdouts in CISP. Test scenes are never published. Weekly rotation prevents memorization. Prospective testing on synthetically generated scenes with known ground truth.

### Failure Mode 4: Platform Lock-Out

**Risk**: PWM becomes a closed platform that excludes smaller labs or non-commercial researchers.

**Countermeasure**: Open-source core evaluation protocol. Free tier for academic submissions to CISP. Community governance with rotating stewards. Replication Packs are always public.

### Failure Mode 5: Relevance Drift

**Risk**: New imaging modalities emerge (quantum imaging, neuromorphic sensors) that PWM doesn't cover.

**Countermeasure**: The PhysicsOperator protocol is extensible by design. Community can contribute new modalities via YAML registry entries. The flywheel's reinvestment stage explicitly funds new modality development. Target: +10 modalities per quarter.

---

## Part 7: Economic Model

### Revenue Streams (Rail Economics)

| Revenue Stream | Model | Margin Profile |
|---------------|-------|---------------|
| **CISP Submission Fees** | Per-submission or annual subscription | Low revenue, high network effect |
| **Calibration-as-a-Service** | Pay-per-calibration ($/scene) | High margin, scales with compute cost reduction |
| **Compute Escrow Management** | % of escrowed funds | Financial services margin |
| **Imaging SLAs** | Outcome-based contracts ($/validated-reconstruction) | Performance-linked, high accountability |
| **Enterprise Evaluation Licenses** | Annual license for private CISP tracks | Recurring, high switching cost |
| **Data Trust Stewardship** | Per-query access to Physics Imaging Data Bank | Scales with corpus size |

### The New Economic Dashboard (PWM Edition)

From solveeverything.org's "new economic dashboard," adapted for PWM:

| Metric | Definition | Current | Target (18 months) |
|--------|-----------|---------|-------------------|
| **RoCS** (Return on Cognitive Spend) | dB improvement per dollar of compute | 3.01 dB / ~$0.50 = 6.02 dB/$ | > 20 dB/$ |
| **D2R** (Design-to-Reconstruction) | Seconds from spec to validated result | 484 sec | < 60 sec |
| **E2C Index** (Energy-to-Compute) | Validated reconstructions per kWh | ~7 recon/kWh | > 100 recon/kWh |
| **Modality Coverage** | Fraction of known imaging physics | 64/~100 = 64% | > 85% |
| **Community Adoption** | Labs using PWM evaluation protocol | 1 (internal) | > 20 |

---

## Part 8: The Strategic Argument

### Why Now (The Foundry Window)

The solveeverything.org framework states:

> "Within the next 18 months, that metal will cool and harden. The decisions we make today regarding technical standards, data rights, and supply chains will set path dependencies: permanent tracks that will guide or constrain the economy for decades."

For computational imaging, the standards are **not yet set**. There is no CASP equivalent. There is no Protein Data Bank equivalent. There is no universal evaluation protocol. The field is fragmented: every lab uses different metrics, different datasets, different mismatch models.

**PWM is positioned to be the first credible, comprehensive, standardized evaluation infrastructure for computational imaging.** The 4-scenario protocol, the mask-sensitivity spectrum, the 5-method comparison, and the 64-modality coverage are unmatched.

The lock-in window is open. The first standard that 5+ labs adopt becomes permanent.

### The One-Line Pitch

> PWM is not a moonshot — it is the **targeting system** that aims moonshots at the right problems across all imaging-dependent domains, and the **action network** that translates their digital solutions into calibrated physical measurements.

### The solveeverything.org Integration

PWM should be proposed to solveeverything.org not as a new moonshot, but as a **cross-cutting infrastructure primitive** that enables existing moonshots:

- Listed under "Action Networks" and "Targeting Authorities"
- Positioned as the "CASP for Imaging" — a shared evaluation infrastructure
- Provides the PhysicsOperator protocol as a standard API surface
- Hosts CISP as a public, adversarial, rolling competition

This is how PWM becomes a rail that every train must ride on.

---

## Appendix: Current PWM Assets for Lock-In

### Already Built (Lock-In Ready)

- 64 imaging modalities under one PhysicsOperator protocol
- 89 graph templates with YAML registry
- 4-scenario evaluation protocol (Ideal, Assumed, Corrected, Oracle)
- 5-method benchmarking (GAP-TV, MST-S, MST-L, HDNet, PnP-HSICNN)
- Mask-sensitivity spectrum characterization
- Self-supervised calibration pipeline (two-stage differentiable)
- 2904 tests passing, 0 failures
- Replication Packs (JSON + scripts + configs)
- DR-AIS compatible YAML registries

### Validated Results (Proof of Concept)

- CASSI: 10-scene KAIST, 5 methods, 4 scenarios, +3.01 dB calibration gain
- SPC: 4 methods, 3 scenarios, ADMM 27.52 dB
- Parameter recovery: 5-parameter mismatch, RMSE < 1 px
- Timing: 484 ± 45 sec/scene end-to-end

### Gap Analysis (Must Build for Lock-In)

| Gap | Priority | Effort | Impact |
|-----|---------|--------|--------|
| Physics Imaging Data Bank (1000+ scenes) | Critical | High | Observability layer |
| CISP public competition platform | Critical | High | Targeting system |
| Open-source `pwm-eval` package | High | Medium | Adoption catalyst |
| External lab validation (5+ labs) | High | Medium | Network effect trigger |
| Compute Escrow smart contracts | Medium | Medium | Financial primitive |
| Robotic lab API integration | Medium | High | Action surface |
| Multi-objective Pareto scoring | Medium | Low | Anti-gaming |
