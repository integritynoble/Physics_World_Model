# Nature Paper Feasibility Assessment: Agent-Based Imaging System Design

## Paper Concept

**Core Claim:** LLM agents (Plan / Judge / Performance) composing from a proven-complete
11-primitive basis can design *any* imaging system, and the designed system has provably
small error compared to real application systems.

**Building On:** "Eleven Primitives and Three Gates: The Universal Structure of
Computational Imaging" (flagship paper), which proves that 11 physical primitives
(Propagate, Modulate, Project, Encode, Convolve, Accumulate, Detect, Sample, Disperse,
Scatter, Transform) with three diagnostic gates can express any imaging forward model
with representation error epsilon < 0.01 across 168+ modalities and 5 carrier families.

---

## What Is Strong

### 1. Foundation Is Rock-Solid

The flagship paper provides a rigorous mathematical basis:

- **Finite Primitive Basis Theorem** with epsilon < 0.01 approximation guarantee
- Validated on **12 real modalities** across **5 carrier families** (optical, X-ray,
  electron, spin, acoustic)
- **168+ modality registry** with 37 unique DAG patterns
- **Minimality proof** — removing any primitive breaks coverage
- **Basis saturation** — no new primitive needed after N = 35 modalities

### 2. Hot Topic — AI Designs Physical Systems

Nature has published paradigm-level AI-for-science papers:

| Paper | Domain | Impact |
|-------|--------|--------|
| AlphaFold | Protein structure | Predicted any protein fold |
| GNoME | Materials discovery | 2.2M new stable crystals |
| **This paper** | **Imaging systems** | **Design any imaging system from prompts** |

The analogy is direct: AlphaFold navigates the space of protein folds;
this work navigates the space of imaging forward models via a complete primitive basis.

### 3. Clean Logical Chain

```
11 primitives are epsilon-complete (Theorem 1, proven in flagship)
    --> Agents compose ONLY from these primitives (constrained generation)
    --> Any agent output is a valid primitive DAG
    --> Error bound from Theorem 1 applies automatically
    --> Designed system approximates real system within epsilon
```

### 4. Three-Agent Pipeline Is Novel

No prior work uses multi-agent LLM systems for physics-grounded imaging design:

- **Plan Agent** — generates structured spec from natural language + database context
- **Judge Agent** — validates physical/algorithmic feasibility (passes/rejects with feedback)
- **Performance Agent** — simulates and quantifies expected metrics
- **Refinement loop** — iterative improvement until judge passes (up to 3 rounds)

---

## Critical Gaps (Must Fix Before Submission)

### Gap 1: Agents Don't Enforce the Primitive Basis

**Problem:** The Plan Agent is an LLM prompt that outputs free-form JSON. There is
**no formal guarantee** that the output is a valid composition of the 11 primitives.
The flagship paper proves representation completeness of the *basis*, not that an
*LLM* will correctly use it.

**Current state:**
```
User prompt --> Gemini 2.5 Flash --> free-form JSON --> no validation against primitives
```

**What is needed:**
```
User prompt --> LLM proposes --> Primitive Compiler validates DAG --> reject if invalid
                                      |
                              Only allows: P, M, Pi, F, C, Sigma, D, S, W, R, Lambda
                              Checks: typed edges, parameter bounds, adjoint consistency
```

**Deliverable:** A **Constrained Primitive Compiler** that:
1. Parses agent output into a typed DAG over the 11 primitives
2. Validates edge types (each primitive has defined input/output types)
3. Checks parameter bounds (e.g., Detect restricted to 5 families with <= 2 params)
4. Rejects any spec not expressible as a valid primitive composition
5. When valid, the epsilon < 0.01 bound from Theorem 1 applies automatically

### Gap 2: No Real-Hardware Validation of Agent-Designed Systems

**Problem:** The flagship paper validates primitives on 12 real systems. But the agent
pipeline has only been tested with **simulated demos**. For Nature, the agent must
design systems that are compared against known real systems.

**What is needed (minimum 10 modalities, 3+ carrier families):**

| # | Modality | Carrier | Ground-Truth Source | Validation |
|---|----------|---------|-------------------|------------|
| 1 | CT (sparse-view) | X-ray | LoDoPaB-CT real sinograms | Compare H_agent vs H_real |
| 2 | MRI (parallel imaging) | Spin | M4Raw 8-channel k-space | Compare forward DAG |
| 3 | CASSI | Optical | Real CASSI mask + dispersion | Compare sensing matrix |
| 4 | CACTI | Optical | Real coded aperture video | Compare temporal encoding |
| 5 | Ultrasound (plane wave) | Acoustic | PICMUS RF data | Compare delay model |
| 6 | Fluorescence microscopy | Optical | Real PSF + camera data | Compare PSF + noise |
| 7 | Lensless imaging | Optical | Real PSF calibration | Compare forward operator |
| 8 | Holography | Optical (coherent) | Real interference pattern | Compare propagation |
| 9 | Cryo-EM | Electron | EMDB structures | Compare CTF model |
| 10 | OCT | Optical (coherent) | Real A-scan data | Compare interference model |

**Metric per modality:**
```
Relative operator error = ||H_agent - H_real|| / ||H_real|| < epsilon
```

**Protocol (4-scenario from flagship):**
1. Scenario I: True operator H_real on real data --> baseline PSNR_I
2. Scenario II: Agent operator H_agent on real data --> PSNR_II (mismatch)
3. Scenario III: Oracle correction --> PSNR_III (upper bound)
4. Scenario IV: Agent + auto-correction --> PSNR_IV (practical result)

Recovery ratio: rho = (PSNR_IV - PSNR_II) / (PSNR_I - PSNR_II)
Target: rho > 0.85 across all modalities

### Gap 3: Agent Contribution Beyond Prompting

**Problem:** Currently Plan Agent = LLM prompt --> JSON. This is engineering, not
science. Nature requires demonstrable scientific contribution.

**What is needed — Ablation study:**

| Configuration | Description | Expected Result |
|--------------|-------------|-----------------|
| A: Raw LLM | Single prompt, no primitives, no judge | Baseline (likely fails physics) |
| B: Constrained LLM | Single prompt + primitive compiler | Better (valid DAGs, some errors) |
| C: Plan + Judge | Plan agent + judge validation | Better (catches infeasible designs) |
| D: Plan + Judge + Refine | Full loop with refinement | Best (iterative improvement) |
| E: Plan + Judge + Refine + Perf | Full pipeline | Best + quantified metrics |

**Key metrics per configuration:**
- DAG validity rate (% of outputs that are valid primitive compositions)
- Physical correctness (% passing expert review)
- Operator error vs. real system
- Number of refinement rounds to convergence

**Convergence analysis:**
- Plot operator error vs. refinement round (should decrease monotonically)
- Show that judge feedback is specific and actionable (not generic)
- Compare judge pass rate across modalities

### Gap 4: Error Bound Proof Has a Logical Gap

**Problem:** The claim chain has an unproven middle step:

```
Step 1: 11 primitives are epsilon-complete        [PROVEN in flagship Theorem 1]
Step 2: Agents compose from these primitives       [NOT PROVEN -- agents output free text]
Step 3: Designed system ~= real system             [FOLLOWS from Steps 1+2]
```

**Two paths to close this gap:**

**Path A — Formal proof (stronger, preferred):**
1. Define a formal grammar G over the 11 primitives
2. Show the Constrained Primitive Compiler accepts only strings in L(G)
3. Therefore any accepted agent output is a valid primitive DAG
4. By Theorem 1, representation error < epsilon
5. QED: agent-designed system approximates real system within epsilon

**Path B — Empirical completeness test (weaker but sufficient):**
1. Run agent on N >= 50 diverse modalities (all 168 in registry if possible)
2. Show 100% of outputs pass the Primitive Compiler
3. For the 10+ modalities with real data, show operator error < epsilon
4. Statistical argument: zero failures in N trials gives confidence >= 1 - 1/N

**Recommended: Do both.** Formal proof for the compiler, empirical test for the agent.

---

## Recommended Paper Structure

### Title (Options)

1. "Agent-Designed Imaging: Provably Faithful System Design via Compositional Physical Primitives"
2. "From Prompt to Physics: Multi-Agent Design of Imaging Systems with Guaranteed Fidelity"
3. "Designing Any Imaging System: LLM Agents over a Complete Physical Primitive Basis"

### Abstract (Draft)

> Designing imaging systems requires deep expertise across physics, engineering, and
> signal processing — expertise that is siloed by modality and institution. We show
> that a multi-agent AI pipeline (Plan, Judge, Performance) can design physically
> faithful imaging systems for *any* modality by composing from a proven-complete basis
> of 11 physical primitives. Because the agent output is constrained to valid primitive
> compositions, the epsilon < 0.01 approximation guarantee from the Finite Primitive
> Basis Theorem applies automatically: every agent-designed forward model approximates
> the true physical operator within the representation error bound. We validate this
> on N modalities across K carrier families, comparing agent-designed systems against
> real hardware measurements, achieving recovery ratios rho > 0.85 in all cases. The
> refinement loop converges in <= 3 rounds for M% of designs. This work demonstrates
> that imaging system design can be reduced to constrained composition over a finite
> physical vocabulary, making expert-level design accessible from natural language.

### Main Sections

```
1. Introduction
   - The design problem: modality-siloed, expert-intensive
   - Key insight: 11 primitives + agents = any system
   - Main result: provably faithful design with epsilon guarantee

2. Results
   2.1 Constrained Primitive Compiler
       - Typed DAG grammar over 11 primitives
       - Validation rules (edge types, parameter bounds, adjoint consistency)
       - Acceptance rate: 100% of valid designs, 0% of hallucinated physics

   2.2 Agent-Designed vs. Real Systems (The Money Figure)
       - Table: 10+ modalities, operator error, recovery ratio
       - Figure: prompt --> DAG --> comparison with real DAG --> error < epsilon

   2.3 Multi-Round Refinement Convergence
       - Operator error decreases with refinement rounds
       - Judge feedback specificity analysis
       - Convergence in <= 3 rounds for X% of modalities

   2.4 Ablation: Why Agents + Constraints + Refinement
       - Raw LLM vs. constrained vs. full pipeline
       - DAG validity rate, physical correctness, operator error
       - Each component provides measurable improvement

   2.5 Generalization to Unseen Modalities
       - Hold out K modalities from agent training context
       - Agent designs them from first principles + primitive library
       - Error still within epsilon

3. Discussion
   - Design as finite search over physical operators
   - Relationship to AlphaFold (fold space) and GNoME (crystal space)
   - Limitations: multi-parameter mismatch, nonlinear regimes
   - Future: hardware-in-the-loop, automated fabrication

4. Methods
   4.1 Eleven Primitive Basis (summary from flagship, full in Supplementary)
   4.2 Agent Architecture (Plan / Judge / Performance)
   4.3 Constrained Primitive Compiler (formal grammar)
   4.4 Validation Protocol (4-scenario, recovery ratio)
   4.5 Error Bound Derivation
```

### Key Figures

**Figure 1 — Overview:**
```
Natural language prompt
    |
    v
[Plan Agent] --> proposed DAG
    |
    v
[Primitive Compiler] --> validated 11-primitive DAG
    |
    v
[Judge Agent] --> feasible? --NO--> [feedback] --> [Plan Agent] (loop)
    |YES
    v
[Performance Agent] --> metrics (PSNR, SSIM, SNR)
    |
    v
Compare: ||H_agent - H_real|| / ||H_real|| < epsilon
```

**Figure 2 — The Money Figure:**
Side-by-side comparison for 10 modalities:
- Left column: Agent-designed DAG (auto-generated from prompt)
- Right column: Ground-truth DAG (from flagship paper)
- Color-coded: green = matching primitive, yellow = parameter mismatch, red = missing
- Bar chart: operator error for each modality (all below epsilon line)

**Figure 3 — Convergence:**
- X-axis: refinement round (1, 2, 3)
- Y-axis: operator error (decreasing)
- One curve per modality, all converging below epsilon

**Figure 4 — Ablation:**
- Grouped bar chart: 5 configurations x 10 modalities
- Shows progressive improvement from raw LLM to full pipeline

**Figure 5 — Cross-Carrier Universality:**
- Radar/spider plot: 5 carrier families (optical, X-ray, electron, spin, acoustic)
- Each axis: recovery ratio (all > 0.85)
- Demonstrates carrier-agnostic design capability

---

## Venue Recommendation

| Venue | Feasibility | Requirements |
|-------|------------|--------------|
| **Nature** (flagship) | Hard but possible | All 4 gaps fixed, 10+ real-hardware validations, paradigm framing |
| **Nature Machine Intelligence** | Realistic | Gaps 1-3 fixed, 5+ modality validations |
| **Nature Methods** | Most realistic | Gaps 1-2 fixed, demonstrate utility for 5+ modalities |
| **Nature Computational Science** | Realistic | Gaps 1-3 fixed, emphasis on computational framework |

**For Nature flagship**, the framing must be paradigm-level:
> "We show that imaging system design — historically requiring years of domain expertise —
> can be reduced to constrained composition over 11 physical primitives, making any imaging
> system designable from natural language with provable physical fidelity."

---

## Implementation Roadmap

### Phase 1: Constrained Primitive Compiler (Weeks 1-3)

- [ ] Define formal grammar G over 11 primitives
- [ ] Implement typed DAG validator (edge types, parameter bounds)
- [ ] Add adjoint consistency test (randomized dot-product, delta < 1e-6)
- [ ] Integrate into Plan Agent output pipeline
- [ ] Test: 168 modalities from registry, 100% acceptance for valid designs

### Phase 2: Real-Hardware Validation (Weeks 4-8)

- [ ] Select 10 modalities with available real data
- [ ] For each: run agent design from prompt, extract DAG
- [ ] Compare agent DAG vs. ground-truth DAG (operator error metric)
- [ ] Run 4-scenario protocol (ideal/mismatch/oracle/corrected)
- [ ] Compute recovery ratios, compile results table

### Phase 3: Ablation & Convergence Analysis (Weeks 6-8)

- [ ] Implement 5 ablation configurations (A through E)
- [ ] Run each on all 10 validation modalities
- [ ] Record: DAG validity, physical correctness, operator error
- [ ] Analyze refinement convergence curves
- [ ] Statistical tests (paired t-test, bootstrap CI)

### Phase 4: Paper Writing (Weeks 8-12)

- [ ] Draft main text (3000 words for Nature)
- [ ] Create 5 main figures + Extended Data
- [ ] Write Methods section
- [ ] Supplementary: full primitive grammar, all 168 modality results, proofs
- [ ] Internal review, revisions

### Phase 5: Submission Preparation (Weeks 12-14)

- [ ] Cover letter emphasizing paradigm shift
- [ ] Suggest reviewers (computational imaging + AI-for-science)
- [ ] Prepare response template for likely reviewer concerns
- [ ] Submit

---

## Likely Reviewer Concerns (and Preemptive Responses)

### Concern 1: "The agents just call an LLM — where's the science?"

**Response:** The science is in the *constraint*. An unconstrained LLM can hallucinate
non-physical systems. Our Constrained Primitive Compiler guarantees that every output
is a valid composition of 11 physically grounded primitives, which — by Theorem 1
of [flagship] — approximates any real imaging operator within epsilon < 0.01. The agents
provide the generation; the compiler provides the guarantee.

### Concern 2: "How do you know the error bound holds for agent outputs?"

**Response:** Two-level guarantee. (1) Formal: the Primitive Compiler accepts only
valid typed DAGs over the 11 primitives; by Theorem 1, any such DAG has representation
error < epsilon. (2) Empirical: we validated on N modalities with real data, measuring
||H_agent - H_real||/||H_real|| < epsilon in all cases.

### Concern 3: "This only works for imaging systems you've already catalogued."

**Response:** The held-out test (Section 2.5) demonstrates generalization. K modalities
were excluded from the agent's database context, yet the agent correctly designed them
from the primitive library alone, with error < epsilon. The primitive basis is
*universal* (proven in flagship), so any imaging system expressible as sequential-parallel
composition of bounded operators is within scope.

### Concern 4: "What about nonlinear/complex systems?"

**Response:** The Transform (Lambda) and Detect (D) primitives handle nonlinear physics
(beam hardening, phase wrapping, saturation) via 5 enumerable families each.
We validate on CT (beam hardening), MRI (phase wrapping), and fluorescence (saturation).
Systems requiring fundamentally new nonlinearity trigger the extension protocol
(Section 4.1), which adds primitives via a formal process with backward-compatible
closure testing.

### Concern 5: "The LLM might become outdated — is this reproducible?"

**Response:** The scientific contribution is the *framework* (primitive basis +
constrained compiler + validation protocol), not the specific LLM. Any sufficiently
capable LLM can serve as the generation engine; the compiler guarantees physical
validity regardless. We test with Gemini 2.5 Flash and Claude Opus 4.6 and show
equivalent results (Supplementary Table S-X).

---

## Bottom Line

The **idea** is Nature-worthy: reducing imaging system design to constrained composition
over a proven-complete physical basis, with guaranteed approximation fidelity.

The **execution** requires:
1. A Constrained Primitive Compiler (closes the proof gap)
2. Real-hardware validation on 10+ modalities (closes the evidence gap)
3. Ablation study (closes the contribution gap)
4. Formal error bound derivation (closes the theory gap)

With these four deliverables, the paper has a realistic path to Nature or
Nature Machine Intelligence.
