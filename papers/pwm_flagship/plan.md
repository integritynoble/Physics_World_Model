# Flagship Paper Restructuring Plan (v3)

**Target venue:** Nature
**Date:** 2026-02-21
**Core idea:** Restructure the flagship paper so that the **Finite Primitive Basis Theorem** and the **Triad Decomposition** are the two core theoretical contributions — the paper's reason for existing. Everything else (OperatorGraph IR, autonomous correction, hardware validation) supports and validates these two contributions.

---

## Why restructure?

The current paper reads as a **systems paper**: "Here is PWM, a framework with agents, templates, and correction pipelines." The theoretical contributions (Theorem 1, Triad) are embedded within the systems description and feel like features of the toolkit.

Nature publishes **discoveries**, not toolkits. The restructured paper should read:

> "We prove two fundamental results about computational imaging:
> 1. **Every imaging forward model** in a broad operator class can be decomposed into a DAG of exactly **10 canonical primitives** (Finite Primitive Basis Theorem).
> 2. **Every reconstruction failure** decomposes into exactly **three root causes** — information deficiency, noise, and operator mismatch — with mismatch universally dominant (Triad Decomposition).
>
> These results yield a modality-agnostic framework that diagnoses and corrects imaging failures across 7+ modalities and 5 physical carriers, recovering up to +48 dB of lost quality."

The correction pipeline and hardware experiments become **consequences of the theory**, not the main contribution.

---

## Proposed New Structure

### Title (options)

**Option A (theorem-forward):**
> "Two Fundamental Laws of Computational Imaging: A Finite Primitive Basis for Forward Models and a Triad Decomposition for Reconstruction Failure"

**Option B (shorter):**
> "The Finite Primitive Basis and Triad Decomposition: Universal Laws for Computational Imaging"

**Option C (Nature style — finding-forward):**
> "Ten Primitives and Three Gates: The Universal Structure of Computational Imaging"

**Option D (current title, minimally modified):**
> "Physics World Models: A Finite Primitive Basis Theorem and Triad Decomposition Law for Computational Imaging"

---

### New Section Order

| # | Section | Role | Est. length |
|---|---------|------|-------------|
| 0 | **Abstract** | Lead with the two theorems, then consequences | ~200 words |
| 1 | **Introduction** | The problem: imaging lacks a unifying theory. Two contributions. | ~600 words |
| 2 | **The Finite Primitive Basis** | Core contribution #1: the representation theorem | ~1200 words |
| 3 | **The Triad Decomposition** | Core contribution #2: the diagnostic law | ~800 words |
| 4 | **Consequences: Diagnosis and Correction** | The theory enables a practical framework | ~600 words |
| 5 | **Empirical Validation** | Validates both contributions across modalities | ~1200 words |
| 6 | **Discussion** | Implications, periodic table analogy, limitations | ~600 words |
| — | **Online Methods** | Technical details (keep existing, lightly edited) | separate |
| — | **Supplementary** | Expanded proofs, full tables, hardware details | separate |

**Total main text: ~5200 words** (within Nature's ~5000 word guideline with minor trimming)

---

### Section-by-Section Plan

#### 0. Abstract (~200 words, down from current ~350)

**Current problem:** Abstract is too long and leads with the PWM framework description.

**New structure:**
1. (1 sentence) Computational imaging systems fail because the assumed forward model diverges from true physics — but no unified theory explains why or predicts when.
2. (2 sentences) We prove the **Finite Primitive Basis Theorem**: every imaging forward model in the Tier-2 operator class admits an ε-approximate representation as a DAG over exactly 10 canonical primitives covering propagation, interaction, encoding, and detection.
3. (2 sentences) We establish the **Triad Decomposition**: every reconstruction failure decomposes into three gates — information deficiency, carrier noise, and operator mismatch — with mismatch universally dominant across all validated modalities.
4. (2 sentences) These two results yield a modality-agnostic diagnostic and correction framework. Across seven modalities and five physical carriers, autonomous correction recovers +0.76 to +48.25 dB; hardware validation on real CASSI and CACTI instruments confirms mismatch dominance.
5. (1 sentence) A held-out closure test on 8 additional modalities — including quantum ghost imaging and Compton scatter — confirms basis completeness with sublinear, saturating growth.

---

#### 1. Introduction (~600 words)

**New framing — lead with the theoretical gap:**

**Para 1: The promise and failure of computational imaging** (~120 words)
- Computational imaging extracts more information than classical optics permits
- Community has invested a decade in solver improvements (CS → PnP → deep unrolling → transformers)
- Yet systems routinely fail on real instruments
- The field lacks a theoretical framework to explain WHY

**Para 2: The scale of the problem — a motivating example** (~120 words)
- CASSI example: MST-L achieves 34.81 dB ideal, drops to 20.83 dB with sub-pixel mismatch
- A calibration error erases twice the gains of a decade of solver R&D
- This pattern repeats across modalities (CACTI, MRI, CT)
- Key insight: the community has been optimizing the wrong variable

**Para 3: Two missing theoretical foundations** (~120 words)
- **Missing foundation #1: Representation.** No formal result guarantees that a finite set of primitive operators suffices to represent all imaging forward models. Without this, every new modality requires bespoke engineering.
- **Missing foundation #2: Diagnosis.** No systematic framework decomposes reconstruction failures into root causes. Calibration, noise, and information deficiency are conflated.

**Para 4: Our contributions** (~120 words)
- We prove the **Finite Primitive Basis Theorem** (Theorem 1): 10 primitives suffice for all Tier-2 imaging
- We establish the **Triad Decomposition**: three gates, Gate 3 universally dominant
- These two results are complementary: the Finite Primitive Basis provides a universal representation; the Triad provides a universal diagnostic law over that representation
- Together they yield a practical framework that diagnoses and corrects imaging failures across 7 modalities without modality-specific tuning

**Para 5: Validation summary** (~120 words)
- 7 modalities validated end-to-end (CASSI, CACTI, SPC, Lensless, Ptychography, MRI, CT)
- Hardware validation on real CASSI and CACTI instruments
- Held-out closure test on 8 additional modalities confirms basis completeness
- The basis grows sublinearly and saturates at K=10

---

#### 2. The Finite Primitive Basis (~1200 words)

This is **Core Contribution #1** and gets the most space. Combines material from current "OperatorGraph IR" section with the theorem.

**Para 1: The OperatorGraph representation** (~150 words)
- Every imaging forward model is encoded as a typed DAG
- Each node wraps one primitive operator; edges define data flow
- Both forward() and adjoint() implemented with validated consistency
- Keep brief — the IR is the delivery vehicle for the theorem, not the contribution itself

**Para 2: Ten canonical primitives** (~100 words + table)
- Present the 10-primitive table (current lines 114-132): P, M, Π, F, C, Σ, D, S, W, R
- Detect constraint: 5 families, ≤2 parameters each, not a universal approximator
- This is the primitive library B

**Para 3: Two-layer architecture** (~100 words)
- Layer A: Physics-stage families (propagation, interaction, encoding-projection, detection-readout)
- Layer B: Primitives map to stages
- Keep the existing mapping table (current lines 136-137)

**Para 4: Definitions** (~100 words)
- Definition 1: C_Tier2 (Tier-2 operator class)
- Definition 2: ε-approximate representation
- Keep existing definitions (current lines 141-149)

**Para 5: Theorem 1 statement** (~50 words)
- Keep existing theorem (current lines 151-154)

**Para 6: Proof sketch** (~150 words)
- 4-phase decomposition (propagation → interaction → encoding → detection)
- Sub-multiplicativity error bound
- Reference to Supplementary Note and companion paper
- Keep existing proof sketch (current lines 156), possibly expand slightly

**Para 7: Scope box** (~80 words)
- What C_Tier2 covers and excludes
- Keep existing scope box (current lines 160-161)

**Para 8: Extension protocol** (~100 words)
- When new primitives are warranted
- Worked example: Compton → Scatter
- Keep existing (current lines 163-164)

**Para 9: Empirical validation of the theorem** (~200 words)
- **Closure test** (MOVE from current Results section to here): 8 held-out modalities, 4-metric cards
- Include the closure test table (current lines 237-254)
- Ghost imaging = operator-equivalent to SPC
- THz = coherent Detect
- Compton → Scatter (only new primitive needed)

**Para 10: Basis-growth saturation** (~100 words)
- Move from Results section
- K=10 at N=31+, sublinear and saturating
- Consistent with theorem prediction
- Figure 9 reference

**Para 11: Physics Fidelity Ladder** (~70 words)
- 4-tier ladder (shift-invariant → shift-variant → nonlinear → full-wave)
- The theorem applies to Tiers 1-2; Tiers 3-4 handled by refinement sub-DAGs

---

#### 3. The Triad Decomposition (~800 words)

This is **Core Contribution #2**. Mostly the current Triad section, tightened.

**Para 1: Three gates** (~200 words)
- Gate 1: Recoverability (null space, information deficiency)
- Gate 2: Carrier Budget (SNR, noise floor)
- Gate 3: Operator Mismatch (H_nom ≠ H_true)
- Keep existing gate descriptions but tighten each to ~60 words

**Para 2: Mathematical formulation** (~100 words)
- 4-Scenario Protocol as the experimental instantiation
- Recovery ratio ρ definition
- Clarify: the Triad is the tripartite decomposition; the 4-Scenario Protocol is its measurement protocol

**Para 3: TriadReport** (~50 words)
- Structured diagnostic output
- Keep brief

**Para 4: Key finding — Gate 3 dominates** (~200 words)
- Across all 9 correction configurations, Gate 3 is dominant
- Theoretical justification (Proposition 2 from Supplementary Note 1)
- The field has been optimizing the wrong variable
- CASSI example: 13.98 dB loss from mismatch vs 7 dB gain from a decade of solver R&D

**Para 5: Relationship to the Finite Primitive Basis** (~100 words) — NEW
- The OperatorGraph representation makes the Triad operational across modalities
- Gate 3 diagnosis works because the DAG structure localizes mismatch to specific primitive nodes
- The same DAG enables correction by perturbing parameters of identified nodes
- The Triad and the Finite Primitive Basis are complementary: one decomposes the operator, the other decomposes the failure

**Para 6: Falsifiable predictions** (~100 words) — NEW (from old plan Enhancement A)
- Prediction 1: Gate 3 dominates whenever calibration error exceeds noise-equivalent resolution
- Prediction 2: Recovery ratio ρ is bounded by the mismatch subspace coherence with the signal prior
- These make the Triad falsifiable

---

#### 4. Consequences: Diagnosis and Correction (~600 words)

This section replaces the current "Autonomous Diagnosis and Correction" section. It's positioned as a **consequence** of the two theoretical contributions, not an independent contribution.

**Para 1: From theory to practice** (~80 words)
- The two theoretical results directly imply a practical framework
- The OperatorGraph enables modality-agnostic reasoning
- The Triad enables root-cause diagnosis
- No LLM, no learned parameters, fully deterministic

**Para 2: Three diagnostic agents** (~150 words)
- RecoverabilityAgent (Gate 1)
- PhotonAgent (Gate 2)
- MismatchAgent (Gate 3) — most consequential, reflecting Gate 3 dominance
- Keep brief: each gets 2-3 sentences (currently each gets a full paragraph)

**Para 3: Correction pipeline** (~150 words)
- Beam search + gradient refinement
- Operates on forward model, not solver
- Solver-agnostic: any existing solver benefits without modification
- Keep existing Algorithm 1/2 description but compress

**Para 4: 4-Scenario Protocol** (~100 words)
- Scenario I (Ideal), II (Mismatch), III (Corrected), IV (Oracle)
- Enables rigorous quantification
- Move from Triad section to here (it's a measurement protocol, not part of the theoretical law)

**Para 5: Calibration accuracy** (~120 words)
- CASSI 5-parameter mismatch example
- Sub-pixel recovery
- Recovery ratios
- Keep existing (current lines 193-198) but compress

---

#### 5. Empirical Validation (~1200 words)

Combines current Results section (minus closure test and basis-growth, which move to §2).

**Para 1: Overview** (~80 words)
- Two-stage validation: controlled simulation + real hardware
- 7 modalities, 5 carriers, 9 correction configurations

**Para 2: Correction results summary** (~150 words)
- +0.76 to +48.25 dB correction gains
- Gate 3 dominant in every case
- Table reference to Supplementary Table S1
- Compress current "Correction results" paragraph

**Para 3: CASSI deep dive** (~200 words)
- Combine mask-geometry-plus-dispersion mismatch
- Uniform Scenario II collapse (20.83-21.88 dB) regardless of solver
- Operator-driven failure confirmed
- Recovery ratios vary by solver (0.22-0.46)
- Compress from current ~300 words

**Para 4: CACTI and SPC** (~150 words)
- CACTI: 20.58 dB loss, multiplicative error amplification across frames
- SPC: gain drift mismatch, TV-based blind calibration
- Inverse performance-robustness relationship
- Compress from current ~250+200 words

**Para 5: Gate 1 and Gate 2 validation** (~100 words)
- Brief: extreme compression and noise sweeps confirm all three gates are real
- Reference Supplementary Tables S10-S11
- Compress from current ~200 words

**Para 6: Zero-shot generalization** (~80 words)
- Hyperparameters tuned on photon-domain transfer to spin/X-ray
- Carrier-agnostic correction confirmed
- Compress from current ~120 words

**Para 7: Hardware validation — CASSI** (~150 words)
- Measurement residual as ground-truth-free diagnostic
- GAP-TV: 1.8× residual ratio
- HDNet: mask-oblivious (1.0×)
- Simulation-to-hardware gap: pre-existing manufacturing errors
- Compress from current ~250 words

**Para 8: Hardware validation — CACTI** (~100 words)
- 10.4× residual ratio (order of magnitude)
- Temporal compression amplifies errors multiplicatively
- Compress from current ~150 words

**Para 9: Autonomous calibration on real data** (~100 words)
- CASSI: 85% recovery, CACTI: 100% recovery, SPC: 86-92% via TV objective
- Brief — details in supplementary
- Compress from current ~200 words

---

#### 6. Discussion (~600 words)

**Para 1: Central finding** (~100 words)
- Operator mismatch is the dominant bottleneck — not solver weakness, not information deficiency, not noise
- Rebalance effort from solver-centric to operator-centric approaches
- A single calibration step can recover more than years of solver R&D

**Para 2: The periodic table analogy** (~100 words)
- Keep existing (current lines 297)
- Pedagogical, not mathematical

**Para 3: Coverage narrative — exotic modalities** (~100 words)
- Ghost imaging = SPC at Tier-2
- THz = coherent Detect
- Compton → Scatter covers 5+ modalities
- Hallmark of a well-chosen basis

**Para 4: Practical implications** (~80 words)
- Clinical MRI coil sensitivity correction
- Remote sensing atmospheric model errors
- CT QC Copilot (brief mention with Supplementary Note reference)

**Para 5: Limitations** (~120 words)
- Software-simulated perturbations, not physical displacement
- Tier 1-2 models only
- Correction limited to declared parameter family
- CT QC uses simulated fleet
- Trim from current ~250 words

**Para 6: Future directions** (~100 words)
- Hardware-in-the-loop validation
- Real-time adaptive calibration
- Prospective clinical deployment
- Scaling the OperatorGraph library
- Compress from current ~150 words

---

### What Moves Where

| Content | Current Location | New Location | Rationale |
|---------|-----------------|-------------|-----------|
| Closure test table | Results §5 | Finite Primitive Basis §2 | Validates Theorem 1 directly |
| Basis-growth analysis | Results §5 | Finite Primitive Basis §2 | Evidence for the theorem |
| 4-Scenario Protocol | Triad §2 | Consequences §4 | It's a measurement method, not the theoretical law |
| Calibration accuracy | Diagnosis §4 | Consequences §4 | Stays in the practical section |
| Physics Fidelity Ladder | OperatorGraph §3 | Finite Primitive Basis §2 | Defines the scope of C_Tier2 |
| CT QC Copilot | Discussion §6, 1 paragraph | Discussion §6 (compressed) + Supplementary | Not a core contribution; move details to supplementary |
| Broader benchmark | Results §5 | Finite Primitive Basis §2 or Supplementary | The 26-modality registry validates the theorem |
| "Primitive operators" informal paragraph | OperatorGraph §3 | Removed | Redundant with the formal primitive table |

---

### Supplementary Changes

The supplementary structure stays mostly the same but needs:

1. **Fix Note 8 numbering collision** (two "Note 8" sections)
2. **Populate placeholder values** in Tables S12, S13
3. **Add Supplementary Table S14** (closure test details, currently missing)
4. **Sole author** — update to Chengshuai Yang only
5. **Move CT QC Copilot details** — if compressed in main text, ensure full details remain in supplementary

---

### Bibliography Changes

1. **Add HATNet reference** (`qu2024hatnet`) — currently cited but missing from bib
2. **No other bib changes needed**

---

### Figure Changes

1. **Figure 9** — Replace placeholder PDF with real TikZ basis-growth saturation curve
2. **Figure reordering** — May need to reorder figure references to match new section order (figures are at the end of the manuscript, so this is just updating `\label`/`\ref` consistency)
3. **No new figures needed** — existing 9 figures cover all content

---

### Authorship

**Sole author: Chengshuai Yang**

Update in:
- `main.tex`: Remove Xin Yuan, update affiliations, Author Contributions, Competing Interests
- `supplementary.tex`: Update author line
- Move Xin Yuan acknowledgment to Acknowledgements section

---

## Implementation Order

1. **Restructure main.tex** — Reorder sections to new structure (§2 Finite Primitive → §3 Triad → §4 Consequences → §5 Results)
2. **Move closure test + basis-growth** from Results into Finite Primitive Basis section
3. **Move 4-Scenario Protocol** from Triad into Consequences section
4. **Add "Relationship to FPB" paragraph** in Triad section
5. **Add "Falsifiable predictions" paragraph** in Triad section
6. **Compress Results section** — tighten CASSI deep dive, CACTI, SPC, hardware validation
7. **Compress Abstract** — rewrite to lead with two theorems (~200 words)
8. **Rewrite Introduction** — new 5-paragraph structure leading with theoretical gap
9. **Fix sole authorship** everywhere
10. **Fix supplementary issues** (Note 8 collision, placeholders, Table S14, sole author)
11. **Add HATNet bib entry**
12. **Generate real Figure 9 PDF**
13. **Word count check and trimming pass**
14. **Compile all documents** (main, supplementary, companion)
15. **Commit and push**

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Restructuring breaks LaTeX cross-references | Compile after each major move; fix refs incrementally |
| Word count exceeds Nature limit after restructuring | The new structure is inherently more compressed; heavy trimming of Results |
| Reviewer says "the Triad is obvious" | Falsifiable predictions + Proposition 2 (formal Gate 3 dominance condition) counter this |
| Reviewer says "10 primitives is arbitrary" | Extension protocol + Compton worked example + basis-growth saturation counter this |
| Reviewer says "companion paper overlap" | Main paper has compact theorem; companion has full formal proof. Clear division of labor. |

---

## Summary

The restructured paper tells a clean two-theorem story:

1. **Theorem 1 (Finite Primitive Basis):** 10 primitives suffice to represent all Tier-2 imaging forward models — proven constructively, validated on 31+ modalities.
2. **The Triad Decomposition:** Every reconstruction failure has exactly three root causes, with operator mismatch universally dominant — formalized mathematically, validated across 7 modalities and 2 real instruments.

Everything else (OperatorGraph IR, agents, correction pipeline, hardware validation) becomes **evidence for and consequences of** these two fundamental results. This is a Nature paper about discoveries, not a systems paper about a toolkit.
