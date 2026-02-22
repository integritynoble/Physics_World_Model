# PWM Flagship: Nature-Grade Revision Strategy

**Date:** 2026-02-22
**Target:** Nature (main journal) or Nature Computational Science
**Central Claim:** "Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging"

---

## A. Executive Revision Strategy (1-page summary)

### Current State
The manuscript presents two theoretical results (Finite Primitive Basis Theorem + Triad Decomposition) and validates them across 7 modalities with hardware data from 5 instruments. The scope is ambitious — a universal framework for computational imaging — but the paper currently reads as a framework/platform paper rather than a tightly argued scientific contribution with one falsifiable central claim.

### Core Problem
The paper tries to do too many things: prove a theorem, introduce a software platform, validate across modalities, demonstrate correction algorithms, show hardware results, and propose a clinical translation roadmap. Nature needs ONE clear scientific finding with decisive evidence.

### Proposed Strategy: Converge on ONE testable scientific claim

**Selected Headline Claim (see Section E for candidates):**
> *A finite operator algebra of 11 physically typed primitives, combined with a tripartite diagnostic decomposition, is sufficient and minimal for representing and diagnosing every computational imaging forward model — and operator mismatch, not solver design, is the dominant reconstruction bottleneck across all validated modalities.*

This claim is:
- **Testable:** 11 is sufficient (constructive DAGs); 11 is necessary (witness modalities); 3 gates are exhaustive (decomposition bound); Gate 3 dominates (empirical + theoretical condition).
- **Falsifiable:** A modality requiring a 12th primitive, or a regime where Gate 1/2 dominates under standard conditions, would refute it.
- **Practically consequential:** Implies calibration is systematically underinvested; a single correction step outperforms solver upgrades.

### Key Structural Changes
1. **Merge the two results into one narrative:** The 11 primitives *enable* the 3-gate diagnosis. Don't present them as separate contributions competing for space.
2. **Lead with the surprising finding:** "Calibration beats solver upgrades" is the hook. The theorem is the mechanism; the Triad is the diagnostic law; the correction results are the proof.
3. **Cut platform engineering details:** Move agent architecture, YAML registries, contract system, RunBundle schema entirely to Supplementary. The main text should contain physics and evidence, not software.
4. **Strengthen the evidence ladder:** The current 7 validated modalities are strong but the paper needs tighter ablations (why 11, not 10? why 3 gates, not 2?) and clearer failure-mode reporting.

---

## B. Detailed Manuscript Surgery Plan

### Title → keep current, it's strong
"Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging"

### Abstract → rewrite (see Section E)
- Lead with the scientific problem (forward-model mismatch), not the framework
- State both results in one sentence each
- Give the key quantitative finding (correction gains)
- State scope explicitly ("7 modalities, 4 carrier families, 5 real instruments")
- End with the implication, not a feature list

### Introduction (currently ~2 pages) → compress to 1.5 pages
**Current issues:**
- Too many citations in the opening paragraph (8 citations in sentence 2)
- "Lacks two theoretical foundations" is a strong claim that needs to be earned
- The two foundations are presented as equal; they should be integrated
- Too much detail about what PWM *does* before explaining *why*

**Surgery:**
1. Para 1: Problem statement (mismatch degrades reconstruction; 3 sentences, zero jargon)
2. Para 2: Why this matters (mismatch is systematic, not accidental; calibration is underinvested)
3. Para 3: Our contribution (one theorem, one decomposition, one correction framework; testable claim)
4. Para 4: Scope and validation summary (7 modalities, 4 carriers, 5 real instruments)
5. Para 5: "Why this matters beyond computational imaging" (see Section 3 below)

### The Finite Primitive Basis (currently ~2 pages) → compress to 1.5 pages
**Keep:** Theorem statement, 11-primitive table, scope box, closure test table, basis-growth figure
**Cut:** Detailed physics-stage mapping paragraph (→ Supplementary), fidelity-level detail (→ Supplementary), full proof sketch (→ Supplementary, already there)
**Add:** One-sentence intuitive explanation of each primitive (currently too formal for Nature)

### The Triad Decomposition (currently ~1 page) → keep at 1 page
**Keep:** Three gate definitions, Gate 3 dominance theorem statement, relationship to FPB
**Fix:** Gate definitions are currently too formal for the main text. Add one intuitive sentence per gate before the math.
**Add:** A "why exactly 3?" paragraph: information loss, noise, and model error are exhaustive and mutually exclusive in origin.

### Consequences: Diagnosis and Correction (currently ~0.5 page) → expand to 1 page
**Keep:** Agent description (compress to 2 sentences), correction pipeline overview
**Add:** The 4-Scenario Protocol deserves more main-text real estate — it's the evaluation backbone
**Cut:** "No LLM required" — irrelevant for Nature readers

### Empirical Validation (currently ~2.5 pages) → restructure to 2.5 pages
**Current structure:** Simulation results → Modality deep dives → Hardware validation → Autonomous calibration → Sim-to-hardware gap
**Proposed structure:**
1. **Cross-modality correction results** (1 summary table, 1 figure — the centerpiece)
2. **Hardware validation** (5 instruments, key numbers)
3. **Ablations** (why 11? why 3 gates? what if you remove one?)
4. **Failure modes** (where does the framework weaken?)

### Discussion (currently ~2.5 pages) → compress to 1.5 pages
**Cut:** Periodic table analogy (too loose for Nature), detailed ESPIRiT comparison (→ Supplementary), verbose roadmap
**Keep:** Sim-to-hardware gap interpretation, falsifiable predictions, limitations
**Add:** Sharper conclusion — what does this change about how the field should invest?

### Methods (currently ~4 pages) → keep at 4 pages
**This is strong.** Minor changes:
- Move agent system architecture and YAML registry details to Supplementary
- Expand statistical analysis section with effect sizes and confidence intervals
- Add explicit reproducibility details (seeds, hardware, runtime)

---

## C. Figure Redesign Plan

### Main Paper Figures (target: 6–7 figures)

| Fig | Current | Proposed Revision | Priority |
|-----|---------|-------------------|----------|
| 1 | PWM overview pipeline | Keep but simplify; reduce to 3 panels: (a) OperatorGraph DAG, (b) Triad diagnosis, (c) correction result. Remove pipeline boxes. | High |
| 2 | OperatorGraph IR + Fidelity Ladder | **Merge with Fig 1** or move to Supplementary. The fidelity ladder is secondary. | Medium |
| 3 | Triad structure + gate binding | Keep; sharpen the heatmap to show Gate 3 dominance more dramatically | Medium |
| 4 | Correction bar chart + zero-shot | **This is the centerpiece.** Redesign as a single panel showing all 7 modalities side by side: Scenario I, II, III, IV bars grouped by carrier family. Add error bars. | Critical |
| 5 | Deep dives + visual comparison | Keep CASSI and CACTI panels; add CT and MRI panels to show carrier diversity | High |
| 6 | Hardware validation | Keep; add CT sinogram and ptychography panels to show 5 instruments | High |
| 7 | Basis growth | Keep as-is — this is the "periodic table" saturation curve | Low |

### NEW Figure: Ablation figure (Fig 4b or Fig 8)
- Panel (a): What happens when you remove each primitive? (11 bars showing ε_tier increase for witness modality)
- Panel (b): What happens when you remove each gate from the diagnostic? (3 bars showing misdiagnosis rate)
- This directly addresses "why 11?" and "why 3?"

### Supplementary Figures
- Move fidelity-ladder figure to Supplementary
- Add per-scene scatter plots for all modalities (currently only averages reported)
- Add bootstrap CI distributions for recovery ratio

---

## D. Missing Experiments Checklist

### Critical (must have for Nature)

| # | Experiment | Current Status | Gap | Risk |
|---|-----------|---------------|-----|------|
| 1 | **Primitive necessity ablation** — remove each of 11 primitives, show ε_tier > 0.01 for witness modality | Proven in FPT paper (Proposition 1) but NOT shown as a figure/table in flagship | Need to import results or generate figure | Low (data exists) |
| 2 | **Gate exhaustiveness argument** — formal argument that information loss, noise, and model error are exhaustive | Informal in text | Need a 1-paragraph formal argument or proposition | Low |
| 3 | **Error bars / confidence intervals on all correction results** | Bootstrap CI for recovery ratio only | Need per-scene error bars on all PSNR comparisons; effect sizes | Medium |
| 4 | **Failure case documentation** — where does correction fail or underperform? | CASSI 5-param recovery is 22–46% (acknowledged) | Need explicit "boundary conditions" section | Low (data exists) |
| 5 | **Ablation: remove one gate from diagnostic** — show that diagnosis accuracy drops | Not done | Need to run: remove MismatchAgent, show misdiagnosis | Medium |

### Important (strongly recommended)

| # | Experiment | Current Status | Gap | Risk |
|---|-----------|---------------|-----|------|
| 6 | **Nonlinear modality validation** — beam hardening CT or phase-wrapped MRI through full 4-scenario | Acknowledged as limitation; only linear forward models validated | Need at least 1 nonlinear modality through full pipeline | High (requires implementation) |
| 7 | **Scaling analysis** — how does correction cost scale with spatial dimension? | Runtime reported for CASSI only | Need runtime table for all 7 modalities | Low |
| 8 | **Cross-modality transfer ablation** — demonstrate that zero-shot hyperparameter transfer actually works by showing the failure when you DON'T transfer | Claimed but not ablated | Need: modality-specific tuned vs. transferred vs. random | Medium |
| 9 | **Second CT dataset** — current CT validation uses simulated sinograms + real sinograms separately | HTC and walnut real sinograms validated | Could strengthen with a 3rd dataset | Low |

### Nice-to-have (strengthens but not required)

| # | Experiment | Notes |
|---|-----------|-------|
| 10 | Physical mask displacement experiment (CASSI hardware-in-the-loop) | Protocol specified; needs instrument access |
| 11 | Higher-acceleration MRI (R=8) validation | Would strengthen the MRI story |
| 12 | Acoustic modality full validation (ultrasound or photoacoustic) | Currently template-only |
| 13 | Comparison with more specialist methods (beyond ESPIRiT) | e.g., auto-focus CT, ePIE refinement |

---

## E. Draft Rewritten Materials

### E1. Title Options

**Bold:** Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging

**Balanced:** A Finite Operator Basis and Tripartite Failure Decomposition for Computational Imaging

**Conservative:** Operator Mismatch Dominates Reconstruction Failure Across Imaging Modalities: Theory and Cross-Modal Validation

**Recommendation:** Keep the bold title. It's memorable, specific (11, 3), and scientifically precise. Nature editors respond to titles that make a clear structural claim.

### E2. Candidate Headline Claims

**Candidate 1 (structural):**
> Every computational imaging forward model admits an ε-approximate representation over exactly 11 physically typed primitives, and every reconstruction failure decomposes into exactly three root causes — with operator mismatch dominant across all validated modalities.

**Candidate 2 (practical):**
> Correcting the forward model recovers more reconstruction quality than upgrading the solver — a finding explained by a universal 11-primitive operator algebra and a tripartite failure decomposition that identifies calibration as the systematic bottleneck.

**Candidate 3 (falsifiable):**
> The space of imaging forward models has finite structural complexity: 11 canonical primitives are sufficient and necessary for ε-approximate representation, and operator mismatch is the binding reconstruction constraint whenever instruments operate above their information-theoretic and noise floors.

**Selected: Candidate 3** — it is the most scientifically precise, directly falsifiable, and combines both theoretical and empirical claims.

### E3. Draft Abstract (Nature-style, evidence-first)

> Computational imaging systems — from coded-aperture spectral cameras to MRI scanners — routinely underperform because the forward model assumed by the reconstruction algorithm does not match the physics that generated the measurement. Yet the field lacks both a universal representation for forward models and a systematic framework for diagnosing why reconstructions fail. Here we establish both. We prove the Finite Primitive Basis Theorem: every imaging forward model in a broad operator class admits an ε-approximate representation as a directed acyclic graph over exactly 11 physically typed primitives — a library that is both sufficient and minimal. We then prove the Triad Decomposition: every reconstruction failure decomposes into three root causes — information deficiency, carrier noise, and operator mismatch — with a formal condition under which mismatch dominates. Across seven modalities spanning four carrier families (optical photons, X-ray photons, electrons, nuclear spins), we show that correcting the forward model autonomously recovers +0.8 to +10.7 dB of mismatch-induced degradation, often exceeding the gap between classical and state-of-the-art deep-learning solvers operating on the same mismatched operator. Hardware validation on five real instruments confirms that operator mismatch — not information deficiency or noise — is the binding reconstruction bottleneck under standard operating conditions. These results imply that calibration, not solver design, is the systematically underinvested component of computational imaging.

### E4. Draft Intro Opening Paragraph (editor-facing, low jargon)

> Every imaging instrument — whether it captures spectra, spins, X-rays, or sound waves — converts a physical scene into digital measurements through a chain of transformations: a wave propagates, interacts with the object, passes through optics, and is recorded by a detector. Computational imaging reconstructs the scene by inverting this chain. The reconstruction algorithm assumes a mathematical model of the chain (the "forward model"), but this model inevitably diverges from reality: optics shift, detectors drift, and calibration degrades. When the model is wrong, the reconstruction fails — not because the algorithm is weak, but because it is solving the wrong problem. This paper establishes that the space of imaging forward models has a surprisingly simple and universal structure, and that exploiting this structure reveals a systematic, cross-modality diagnostic for reconstruction failure.

### E5. "Why This Matters Beyond Computational Imaging" Paragraph

> The finding that a finite set of typed operators suffices to represent all imaging forward models has implications beyond imaging. Any physical measurement system — seismographs, telescopes, particle detectors, environmental sensor networks — faces the same structural problem: a forward model maps the quantity of interest to observations, and model fidelity limits what can be recovered. The OperatorGraph formalism provides a template for decomposing forward models in other domains into typed, composable primitives with validated adjoints. The Triad Decomposition — separating information capacity, noise, and model fidelity as independent failure modes — applies wherever an inverse problem is solved computationally. The practical lesson is domain-general: before investing in a better solver, diagnose whether the model is the bottleneck.

### E6. Draft Conclusion

> We have established that the space of computational imaging forward models has finite structural complexity: 11 physically typed primitives are sufficient and necessary for ε-approximate representation across all clinical, scientific, and industrial modalities. The Triad Decomposition provides a complementary diagnostic law: every reconstruction failure decomposes into information deficiency, carrier noise, and operator mismatch, with mismatch dominant under standard operating conditions across all seven validated modalities and five real instruments. The practical consequence is immediate — a single forward-model correction step, requiring no solver modification or retraining, recovers +0.8 to +10.7 dB of mismatch-induced degradation across four carrier families.
>
> Several limitations qualify these findings. First, the correction pipeline has been validated only on linear forward models; extension to nonlinear modalities (beam hardening CT, phase-wrapped MRI) requires additional validation. Second, the 11-primitive basis is formally universal but empirically validated on 31 modalities; a modality requiring a 12th primitive would trigger the extension protocol, not invalidate the framework. Third, multi-parameter correction (CASSI 5-parameter mismatch) achieves moderate recovery (22–46%), indicating that high-dimensional mismatch manifolds remain challenging.
>
> The framework generates falsifiable predictions: structured illumination microscopy should show +3–8 dB correction gain from pattern phase refinement; OCT should show +2–5 dB from dispersion correction; and any modality where Gate 1 or Gate 2 dominates under standard conditions would require revision of the Gate 3 dominance finding. We invite the community to test these predictions using the open-source pipeline.

---

## F. Coauthor Invitation Template

### Template

> **Subject:** Invitation to contribute to "Eleven Primitives and Three Gates" — a universal framework for computational imaging (Nature submission)
>
> Dear Professor [Name],
>
> We are preparing a manuscript for Nature that establishes a universal structural theory for computational imaging: every imaging forward model admits an ε-approximate representation over exactly 11 physically typed primitives, and every reconstruction failure decomposes into three root causes (information deficiency, noise, operator mismatch) — with mismatch dominant across all validated modalities.
>
> **Current evidence:** 7 modalities validated (CASSI, CACTI, SPC, lensless, CT, ptychography, MRI), 4 carrier families, 5 real instruments, +0.8 to +10.7 dB autonomous correction gains. The FPB Theorem is proven with constructive DAGs for 31 modalities.
>
> **Your contribution:** We would value your expertise in [SPECIFIC AREA — see below]. Concretely, we are seeking [SPECIFIC DELIVERABLE]. We believe you are uniquely suited because [SPECIFIC REASON].
>
> **Timeline:** We aim to submit by [DATE]. The contribution would require approximately [HOURS/WEEKS] of effort.
>
> **Authorship:** All coauthors will meet ICMJE criteria: substantial contribution to conception/design/data/analysis, drafting or critical revision, approval of the final version, and accountability for the work. Author order will reflect contribution magnitude and be agreed transparently before submission.
>
> The open-source codebase is at https://github.com/integritynoble/Physics_World_Model. We are happy to provide a draft manuscript and discuss further.
>
> Best regards,
> Chengshuai Yang

### Suggested Contribution Areas

| Area | Ideal Profile | Specific Deliverable |
|------|--------------|---------------------|
| Theorem rigor / proof refinement | Applied mathematician, operator theory | Review FPB proof; strengthen error bounds; formalize Gate exhaustiveness |
| MRI hardware validation | MRI physicist with scanner access | R=4/R=8 multi-coil experiments with physical coil repositioning |
| CT hardware validation | CT physicist with phantom access | ACR phantom experiments with controlled CoR offset |
| Ptychography validation | Electron microscopist | 4D-STEM experiments with controlled stage drift |
| Statistical methodology | Statistician / ML evaluation expert | Proper CI reporting, multiple comparison corrections, effect size analysis |
| Benchmark design | Inverse problems researcher | Independent evaluation protocol; stress-test the 4-Scenario framework |
| Optical system validation | Experimental optics (CASSI/CACTI hardware) | Physical mask displacement + re-acquisition |

---

## G. Nature Submission Risk Assessment

### Desk-Review Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| "Framework paper, not a scientific finding" | **HIGH** | Lead with the falsifiable claim and the surprising finding (calibration > solver). Minimize platform/software description in main text. |
| "Too broad — covers too many modalities superficially" | Medium | Deepen the evidence for 3–4 flagship modalities (CASSI, CT, MRI, ptychography) rather than spreading thin across 7 |
| "Incremental over the FPT paper" | Medium | Emphasize that FPT provides the *representation*; this paper provides the *diagnostic law* + *empirical validation* + *practical correction*. The Triad is the new scientific contribution. |
| "No real hardware-in-the-loop validation" | Medium | The 5-instrument real-data validation is software-perturbation on real measurements. Acknowledge explicitly; the protocol for physical displacement is specified. |
| Competing interest (C.Y. employed by company developing PWM) | Low | Already disclosed; open-source release mitigates |

### Peer-Review Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| "11 primitives is ad hoc — why not 8 or 15?" | **HIGH** | The necessity proof (Proposition 1 in FPT) is the answer. Import or summarize it prominently. Show the ablation figure. |
| "3 gates is a tautology (information + noise + model = everything)" | **HIGH** | Distinguish: the claim is not that the decomposition exists, but that (a) it is *quantitatively computable*, (b) *Gate 3 dominates* under standard conditions, and (c) the dominance has a *formal condition* (Proposition 2). |
| "Only 7 validated modalities — not universal" | Medium | 31 modalities have constructive DAGs; 7 have full correction validation; 5 have real hardware data. Frame as "validated on 7, formally covers all." |
| "Correction gains are modest for multi-parameter mismatch" | Medium | Acknowledge; 22–46% for CASSI 5-param is honest. Single-parameter correction achieves 85–100%. |
| "No comparison with end-to-end learned approaches" | Medium | The claim is about *calibration*, not *reconstruction*. PWM corrects the operator; any solver (including learned ones) benefits. Show that HDNet/MST-L improve when given the corrected operator. |
| "Statistical rigor — no proper error bars" | Medium | Add bootstrap CIs to all PSNR comparisons; report per-scene results; add effect sizes |

### Mitigation Priority

1. **[Critical]** Reframe as scientific finding, not framework paper — rewrite abstract + intro
2. **[Critical]** Add primitive necessity ablation figure to main text
3. **[Critical]** Add confidence intervals to all quantitative claims
4. **[High]** Strengthen "why 3 gates" argument with formal exhaustiveness proposition
5. **[High]** Move all software engineering details to Supplementary
6. **[Medium]** Add at least 1 nonlinear modality validation (beam hardening CT)
7. **[Medium]** Expand Discussion to explicitly address "is this a tautology?" concern

---

## Supplementary Information Reorganization

### Proposed Structure

| Section | Content | Maps to Main Claim |
|---------|---------|-------------------|
| S1 | Triad Decomposition: mathematical derivations (current Note 1) | Gate formalism |
| S2 | OperatorGraph specification + compilation (current Note 2) | 11 primitives |
| S3 | Primitive necessity proof (import from FPT paper) | Why exactly 11 |
| S4 | Experimental protocol: 4-Scenario details, datasets, solvers | Evaluation rigor |
| S5 | Complete per-scene results: Tables S1–S2, S10–S11 | Evidence depth |
| S6 | 26-modality registry (current Table S3) | Basis coverage |
| S7 | Ablations: gate removal, primitive removal, hyperparameter sensitivity | Necessity arguments |
| S8 | Robustness and failure cases: multi-param mismatch, boundary conditions | Honest limitations |
| S9 | Hardware validation extended: cross-residual analysis (current Note 15) | Real-world evidence |
| S10 | Specialist method comparison: ESPIRiT, auto-focus (current Note S14) | Contextual positioning |
| S11 | Clinical CT QA validation (current Note 7) | Translational value |
| S12 | Reproducibility: RunBundle schema, YAML registries, code pointers | Nature requirements |
| S13 | FPB Theorem expanded proof (current Note 12) | Formal foundations |

---

## Centerpiece Experiment Design (Section 4)

### "The Cross-Modality Mismatch Challenge"

**Concept:** A single figure showing, for each of 7 modalities: (a) reconstruction under perfect calibration, (b) reconstruction under realistic mismatch, (c) PWM-corrected reconstruction, (d) the gap between "best solver on wrong model" vs. "simple solver on corrected model."

**Design:**
- **Modalities:** CASSI (photon), CACTI (photon-temporal), CT (X-ray), MRI (spin), ptychography (electron), SPC (photon-compressed), lensless (photon-incoherent)
- **For each:** Apply the canonical mismatch from mismatch_db.yaml
- **Solvers:** Best available (MST-L for CASSI, EfficientSCI for CACTI, CG-SENSE for MRI, ePIE for ptychography, FBP for CT, FISTA-TV for SPC, Richardson-Lucy for lensless)
- **Baselines:** (i) Ideal operator + best solver (ceiling), (ii) Mismatched operator + best solver (floor), (iii) Mismatched operator + PWM correction + best solver (our method), (iv) Ideal operator + simple solver (GAP-TV/FBP)
- **Key comparison:** Is (iii) > (iv)? If so, "calibration beats solver upgrades" is proven.

**Metrics:** PSNR (primary), SSIM (secondary), per-scene with bootstrap 95% CI

**Ablations:**
- Remove PWM correction → show degradation persists
- Remove one gate from diagnosis → show misdiagnosis
- Use random correction parameters → show that improvement is not trivial

**Falsification:** If for any modality, the best solver on the mismatched operator outperforms the simple solver on the corrected operator, the "calibration > solver" claim fails for that modality. Document this honestly.

**Expected result:** Based on current data, (iii) > (iv) for at least 5 of 7 modalities (CASSI MST-L mismatch drops to 21 dB, well below GAP-TV ideal at 24 dB; similar patterns for CACTI, CT, SPC).

---

## Falsifiability and Rigor (Section 5)

### Hypothesis → Test → Acceptance Criterion → Failure Mode

**H1: 11 primitives are sufficient for all modalities in C_img**
- Test: Constructive DAG for each modality with ε_tier < 0.01
- Acceptance: All 31+ modalities pass
- Failure mode: A modality with ε_tier > 0.01 for all DAGs over B_lib within complexity bounds
- If fails: Extension protocol adds primitive; update |B_lib| = 12

**H2: 11 primitives are necessary (minimal)**
- Test: For each primitive, exhibit witness modality with ε_tier > 0.01 when removed
- Acceptance: All 11 witnesses confirmed
- Failure mode: A primitive whose removal causes no modality to exceed ε
- If fails: Remove that primitive; update |B_lib| = 10

**H3: 3 gates are exhaustive**
- Test: Show that MSE ≤ MSE^(G1) + MSE^(G2) + MSE^(G3) holds for all validated configurations
- Acceptance: Bound holds within 10% for all configurations
- Failure mode: Residual MSE component not attributable to any gate
- If fails: Identify 4th gate (e.g., solver suboptimality) and extend

**H4: Gate 3 dominates under standard operating conditions**
- Test: For each modality, show C_mismatch > max(C_noise, C_recover) under standard parameters
- Acceptance: Gate 3 dominant in ≥ 6 of 7 modalities
- Failure mode: A modality where Gate 1 or 2 dominates at standard compression and SNR
- If fails: Gate 3 dominance is modality-dependent, not universal; weaken claim to "dominant in photon and X-ray modalities"

**H5: Correction recovers ≥ 50% of mismatch gap (single-parameter)**
- Test: Recovery ratio ρ ≥ 0.5 for single-parameter mismatch across all modalities
- Acceptance: Median ρ ≥ 0.5 with 95% CI excluding 0.3
- Failure mode: Systematic ρ < 0.5 for a carrier family
- If fails: Correction algorithm needs modality-specific tuning; weaken "solver-agnostic" claim

### Stress-Testing the Central Claim

**Why exactly 11?**
The FPT paper proves both sufficiency (Theorem 1) and necessity (Proposition 1). The key insight: 11 = 6 physics-stage families × ~2 primitives per family, minus redundancy. The 6 families (propagation, elastic interaction, scattering, pointwise nonlinearity, encoding-projection, detection-readout) are exhaustive because they enumerate all ways a carrier can interact with matter and be measured. Necessity is proven by witness modalities — one per primitive — whose ε_tier exceeds 0.01 when that primitive is removed.

**Why exactly 3 gates?**
The three gates correspond to three independent sources of reconstruction error: (1) information that was never captured (null space), (2) information that was captured but corrupted by noise, (3) information that was captured but misinterpreted due to model error. These are exhaustive because the reconstruction pipeline has only three inputs: the sensing geometry (Gate 1), the carrier statistics (Gate 2), and the forward model (Gate 3). Any reconstruction error must trace to one of these.

**Is 11+3 sufficient?**
For the operator class C_img (finite-stage, bounded regularity, prescribed nonlinearity families): yes, by theorem. For modalities outside C_img (e.g., strongly nonlinear wave propagation without Born approximation, quantum entangled measurement): the extension protocol provides a path, but sufficiency is not guaranteed.

**Falsification path:**
A single modality requiring a 12th primitive, or a single standard-condition regime where Gate 1 or 2 dominates, would falsify the respective claim. The extension protocol and gate-binding analysis are designed to handle such cases constructively.

---

## Reproducibility Package (Section 6)

### Data Availability Statement (journal-style)

> All synthetic measurement data can be regenerated from the OperatorGraph templates and mismatch parameters in the Supplementary Information. The KAIST hyperspectral dataset is publicly available (ref). TSA real-data scenes are from ref. CACTI real-data scenes and masks are from the EfficientSCI repository (ref). CT sinograms are from the FIPS walnut dataset (Zenodo 1254206) and Helsinki Tomography Challenge 2022 (Zenodo 6984868). 4D-STEM ptychography data (SrTiO₃ [001]) is from Zenodo 5113449. Multi-coil MRI k-space data (M4Raw) is from Zenodo 8056074.

### Code Availability Statement (journal-style)

> The complete PWM framework — including OperatorGraph compiler, agent implementations, correction algorithms, YAML registries, evaluation scripts, and RunBundle manifests for all reported experiments — is available at https://github.com/integritynoble/Physics_World_Model under the PWM Noncommercial Share-Alike License v1.0. The codebase requires Python ≥ 3.9, PyTorch ≥ 1.12, and CUDA ≥ 11.3. Exact package versions, random seeds, and GPU specifications for each experiment are recorded in the RunBundle manifests.

### Reproducibility Checklist

| Item | Status | Location |
|------|--------|----------|
| OS, Python, CUDA versions | ✓ | RunBundle manifests |
| Random seeds | ✓ | RunBundle manifests |
| All YAML configs | ✓ | `packages/pwm_core/contrib/` |
| Scripts to reproduce each figure | Partial | Need: `scripts/reproduce_fig{1..7}.py` |
| Datasets with download links | ✓ | Methods section |
| Trained model checkpoints | ✓ (MST-L, HDNet, EfficientSCI from original authors) | README |
| Hardware specs (GPU model, memory) | ✓ | RunBundle manifests |
| Runtime per experiment | Partial | Need table |
| Version tag / commit hash | ✓ | RunBundle manifests |
| Supplement-to-code cross-reference | Missing | Need: Table mapping each Supp. table to script |

---

## Summary: Top 10 Actions by Priority

1. **Rewrite abstract and intro** to lead with the scientific finding, not the framework
2. **Add primitive necessity ablation** as a main-text figure
3. **Add confidence intervals** to all quantitative claims (bootstrap CIs)
4. **Move software engineering** (agents, YAML, contracts, RunBundle) to Supplementary
5. **Strengthen "why 3 gates"** with a formal exhaustiveness argument
6. **Redesign centerpiece figure** (cross-modality correction comparison)
7. **Add explicit failure-mode section** documenting where the framework weakens
8. **Add per-scene scatter plots** to supplement (currently only averages)
9. **Create reproduce scripts** for each main figure
10. **Compress Discussion** — cut periodic table analogy, shorten roadmap, sharpen limitations
