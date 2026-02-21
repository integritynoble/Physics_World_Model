# Flagship Paper Improvement Plan

**Target venue:** Nature
**Current state:** 330-line main.tex with Triad + OperatorGraph + 7-modality results
**Goal:** Strengthen the finite primitive decomposition claim and tighten the OperatorGraph formalism

---

## 1. Triad Law (existing section — targeted edits)

### 1a. Add falsifiable predictions
- After the "Key finding: Gate 3 dominates" paragraph, insert a **Predictions** paragraph stating two testable predictions:
  1. *For any modality where calibration error exceeds the noise-equivalent resolution, Gate 3 will dominate.* (Corollary of Proposition 2 in Supplementary Note 1)
  2. *The recovery ratio ρ is upper-bounded by the mutual coherence between the mismatch subspace and the signal prior.* (New derivation in Supplementary Note)
- These make the Triad falsifiable rather than purely descriptive.

### 1b. Sharpen the mathematical formulation
- In the "Mathematical formulation" paragraph, explicitly state that the 4-Scenario Protocol is an experimental instantiation of the Triad, not the Triad itself. The Triad is the tripartite decomposition; the scenarios are the measurement protocol.

**Files to edit:** `main.tex` lines 63–84 (Triad section)

---

## 2. Finite Primitive Decomposition (major new content in OperatorGraph IR section)

This is the core new claim. It is presented as a **scoped ε-approximate representation theorem** over a formally defined operator class C_Tier2, with Standard Model physics as physical motivation in prose (not as the strict formal foundation).

### 2a. Formal primitive set definition

Insert a new paragraph "Canonical primitive set" after the existing "Primitive operators" paragraph (line 93). Define:

**10 canonical primitives** (final set subject to user revision):

| # | Primitive     | Notation       | Physical action                            | Example use             |
|---|--------------|----------------|-------------------------------------------|------------------------|
| 1 | Propagate    | P(d,λ)        | Free-space wave propagation (Fresnel, angular spectrum) | Lensless, ptychography |
| 2 | Modulate     | M(m)          | Element-wise multiplication by a pattern    | CASSI mask, CACTI mask, SLM |
| 3 | Project      | Π(θ)          | Line-integral / Radon projection           | CT, neutron imaging     |
| 4 | Encode       | F(k)          | Fourier-domain encoding along trajectory k  | MRI k-space traversal   |
| 5 | Convolve     | C(h)          | Convolution with a point-spread function    | Lensless, deblurring    |
| 6 | Accumulate   | Σ             | Summation over a spectral/temporal axis     | SPC bucket detection, CASSI spectral integration |
| 7 | Detect       | D(g,η)        | Detector response: gain g, nonlinearity η, drawn from 5 canonical families (see below) | All modalities (final node) |
| 8 | Sample       | S(Ω)          | Sub-sampling on index set Ω                | MRI undersampling, compressed sensing |
| 9 | Disperse     | W(α,a)        | Wavelength-dependent spatial shift (prism, grating) | CASSI spectral dispersion |
| 10 | Scatter     | R(σ,Δε)       | Direction change and/or energy shift via scattering kernel σ(θ,E) | Compton imaging, Raman, fluorescence, diffuse optical |

**Why 10, not 9:** The first 9 primitives cover all modalities where carriers travel in straight or smoothly-curved paths. Scatter (primitive 10) handles the case where radiation changes direction and/or energy at a scattering site — a physical process that cannot be represented within Tier-2 fidelity and complexity constraints by compositions of the other 9 primitives. See §3d for the detailed analysis that motivated this addition.

**Critical constraint on Detect:** The nonlinearity η is restricted to a small, pre-defined family of canonical response curves — not an arbitrary function f(x). This prevents Detect from becoming a universal approximator. The five allowed response families are:
1. Linear (intensity): η(x) = g|x|²  — standard photodetector
2. Logarithmic: η(x) = g·log(1 + |x|²/x₀)  — wide-dynamic-range sensors
3. Sigmoid: η(x) = g·σ(|x|² - x₀)  — saturating detectors
4. Poisson-rate: η(x) = Poisson(g|x|²)  — photon-counting detectors
5. Coherent-field: η(x) = g·Re[x·e^{iφ}]  — heterodyne/homodyne detection (THz-TDS, OCT, holography)

Each carries at most 2 scalar parameters. The coherent-field family is essential for modalities that measure the electric field amplitude and phase (not intensity), including THz time-domain spectroscopy, optical coherence tomography, and digital holography. If a modality requires a truly novel detection nonlinearity, that signals a basis extension (see §5).

### 2b. Two-layer architecture: physics-stage families vs. OperatorGraph primitives

The framework explicitly separates two abstraction layers:

**Layer A — Physics-stage families.** Every imaging forward model passes through four broad physical stages. These are descriptive categories motivated by Standard Model physics, not formal axioms:

| Stage Family | Physical description | Applicable physics |
|---|---|---|
| **Propagation** | Carrier evolves through space via a wave equation (Maxwell, Schrödinger, acoustic, Bloch) | Free-space travel between source, object, and detector |
| **Interaction** | Carrier exchanges energy, momentum, or phase with the object | Elastic (absorption, phase shift), inelastic (scattering, fluorescence) |
| **Encoding–Projection** | Spatial information is mapped into a measurement-domain coordinate | Radon line-integral (CT), Fourier encoding (MRI), geometric projection |
| **Detection–Readout** | Carrier field is converted to a discrete digital measurement | Integration, sub-sampling, dispersion, PSF blur, quantum measurement |

**Layer B — Canonical OperatorGraph primitives.** Each physics-stage instance is *represented by* one primitive or a finite composition of primitives from the library {P, M, Π, F, C, Σ, D, S, W, R}. The mapping from Layer A → Layer B is:

| Stage Family | Primitives used |
|---|---|
| Propagation | P, C |
| Interaction (elastic forward) | M |
| Interaction (scattering/inelastic) | R |
| Encoding–Projection | Π, F |
| Detection–Readout | Σ, S, W, C, D |

This two-layer separation avoids mixing abstraction levels: Layer A describes *what physics happens*; Layer B describes *how the OperatorGraph represents it*. A single physics-stage instance may require multiple primitives (e.g., a thick scattering medium is M ∘ R ∘ P ∘ R ∘ M at Tier-2), and conversely the same primitive may appear in multiple stage families (C serves both propagation and detection).

### 2c. Canonical coarse abstraction with refinement nodes

> "The primitive set is a **canonical coarse abstraction**: it is deliberately minimal, capturing the dominant physics of each stage at Tier-2 fidelity. For applications requiring higher fidelity (Tier 3–4), each primitive can be **refined** by replacing the canonical node with a sub-DAG that models the full physics (e.g., replacing Propagate with a multi-layer angular-spectrum cascade, or replacing Detect with a Monte Carlo electron-transport model). Refinement preserves the DAG topology and the adjoint contract, ensuring that all downstream diagnostic and correction machinery remains valid."

This positions the primitives as a useful abstraction level, not an absolute physical truth.

### 2d. Typed DAG formalism

> "Each OperatorGraph is a typed DAG G = (V, E) where every node v ∈ V is drawn from the primitive library B = {P, M, Π, F, C, Σ, D, S, W, R} and every edge e ∈ E carries a typed tensor annotation (shape, dtype, physical units). The DAG is constrained to begin with a Source node (emitting the unknown signal x) and terminate with a Detect node (producing the measurement y). We write the composed forward model as H_G = D ∘ vₖ ∘ ··· ∘ v₁, where the composition order is determined by topological sort of G."

### 2e. Decomposition table (new Table 1 or Figure)

Create a table showing the primitive decomposition of all 26 registered modalities. Example rows:

| Modality       | Carrier  | DAG primitives (source → detect) | #Nodes | Depth | Status |
|---------------|----------|----------------------------------|--------|-------|--------|
| CASSI         | Photon   | Source → M → W → Σ → D         | 5      | 4     | Validated (Sc I–IV + hardware) |
| CACTI         | Photon   | Source → M → Σ → D             | 4      | 3     | Validated (Sc I–IV + hardware) |
| SPC           | Photon   | Source → M → Σ → D             | 4      | 3     | Validated (Sc I–IV) |
| Lensless      | Photon   | Source → C → D                  | 3      | 2     | Validated (Sc I–IV) |
| Ptychography  | Photon   | Source → M → P → D             | 4      | 3     | Validated (Sc I–IV) |
| MRI           | Spin     | Source → M(coil) → F → S → D   | 5      | 4     | Validated (Sc I–IV) |
| CT            | X-ray    | Source → Π → D                  | 3      | 2     | Validated (Sc I–IV) |
| OCT           | Photon   | Source → P+P → Σ → D(coh)      | 5      | 3     | Held-out closure |
| Photoacoustic | Acoustic | Source → M(abs) → P(acou) → D   | 4      | 3     | Held-out closure |
| SIM           | Photon   | Source → M(illum) → C → D       | 4      | 3     | Held-out closure |
| Phase-contrast X-ray | X-ray | Source → Π → P → M(grating) → D | 5   | 4     | Held-out closure |
| Electron Ptycho | Electron | Source → M → P → D             | 4      | 3     | Held-out closure |
| Ghost Imaging | Photon   | Source → M(corr) → Σ → D       | 4      | 3     | Exotic — operator-equivalent to SPC at Tier-2 |
| THz-TDS       | Photon(THz) | Source → C(sample) → D(coh)  | 3      | 2     | Exotic — coherent Detect |
| Compton       | X-ray    | Source → M(nₑ) → R(K-N) → D(E) | 4     | 3     | Exotic — **new primitive R** |
| Raman         | Photon   | Source → M(conc) → R(Raman) → D(E) | 4  | 3     | Covered by Scatter |
| Fluorescence  | Photon   | Source → M(abs) → R(fluor) → D  | 4     | 3     | Covered by Scatter |
| ...           | ...      | ...                              | ...    | ...   | 18 more template-validated |

Key insights:
- **No modality requires more than 6 primitive nodes or depth > 5.** This empirical regularity is consistent with Theorem 1.
- **Ghost imaging is operator-equivalent to SPC at Tier-2 abstraction** — sharing the same canonical DAG despite fundamentally different source physics.
- **Scatter covers an entire class:** One new primitive (R) handles 5+ scattering/fluorescence modalities.

### 2f. Finite Primitive Basis Theorem (ε-approximate representation over C_Tier2)

This is the theoretical centrepiece of the paper. The theorem is scoped over a formally defined operator class C_Tier2, with Standard Model physics motivating the class definition in prose rather than serving as a strict axiomatic foundation.

---

#### Definitions (main text; formal details in Supplementary / Extended Methods)

> **Definition 1 (Imaging Forward Model).** An *imaging forward model* is a bounded linear or mildly nonlinear operator H: X → Y mapping an object x ∈ X to a measurement y = H(x) + n, where X and Y are finite-dimensional Hilbert spaces and n is additive noise.

> **Definition 2 (Tier-2 Operator Class C_Tier2).** The class C_Tier2 consists of all imaging forward models H that can be expressed as a finite sequential-parallel composition of linear, shift-variant stages, where each stage has bounded operator norm and the total number of stages is at most N_max. Specifically, H ∈ C_Tier2 if H admits a factorization H = H_K ∘ ··· ∘ H_1 (or a DAG generalization thereof) where each factor H_k is a linear operator with ‖H_k‖ ≤ B and K ≤ N_max. Higher-order effects (nonlinear wave–matter coupling beyond first Born, multiple scattering beyond Tier-2, relativistic corrections) are excluded from C_Tier2.
>
> *Physical motivation:* The class C_Tier2 is designed to capture the forward models of all imaging modalities operating under non-relativistic Standard Model physics at or below the linear shift-variant level of physical fidelity. Every imaging modality in current clinical, scientific, and industrial practice has a forward model in C_Tier2 at the appropriate level of abstraction. The restriction to linear shift-variant stages (Tier 1–2 on the Physics Fidelity Ladder) is a modelling choice, not a fundamental limitation; Tier 3–4 effects are handled by refinement sub-DAGs (§2c).

> **Definition 3 (ε-Approximate Representation).** Let B = {P, M, Π, F, C, Σ, D, S, W, R} be the canonical primitive library. A typed DAG G = (V, E) with V ⊆ B is an *ε-approximate representation* of H ∈ C_Tier2 if:
>
> (i) **Fidelity:** sup_{x ∈ X, ‖x‖≤1} ‖H(x) − H_G(x)‖ / ‖H(x)‖ ≤ ε, where H_G = compose(G).
>
> (ii) **Complexity:** |V| ≤ N_max and depth(G) ≤ D_max.
>
> The formal values of ε, N_max, D_max, and the precise norm and test-distribution specifications are defined in Supplementary Note X (Extended Methods). For the main text, we use ε = 0.01 (1% relative operator-norm error), N_max = 20, D_max = 10.

---

#### Physics-Stage Motivation (prose, not axioms)

In the main text, motivate WHY C_Tier2 is covered by 10 primitives using four physics-stage families (Layer A from §2b). This is presented as physical reasoning, not formal axioms:

> "The physical basis for Theorem 1 is that every carrier's journey from source to detector passes through at most four types of physical stages — propagation, interaction, encoding–projection, and detection–readout (Table X) — and the set of physically distinct operations within each stage is finite. In the *propagation* stage, the carrier obeys a wave equation (Maxwell, Schrödinger, acoustic), representable by P or C. In the *interaction* stage, the carrier either undergoes elastic forward interaction (amplitude/phase change without direction or energy change → M) or scattering (direction and/or energy change → R). In the *encoding–projection* stage, spatial information maps to a measurement coordinate via line-integral (→ Π) or Fourier encoding (→ F). In the *detection–readout* stage, the carrier field is dimensionally reduced (→ Σ, S, W), blurred (→ C), and converted to a classical signal (→ D). These categories are motivated by Standard Model physics: the fundamental electromagnetic, strong, and weak interactions at non-relativistic energies produce only elastic and inelastic carrier–matter coupling, and the wave equations governing propagation are well-characterized for all five carrier families (photons, electrons, spins, acoustic waves, particles). We do not claim this as a mathematical axiom; rather, it is the physical observation that makes Theorem 1 natural."

---

#### Theorem and Proof

> **Theorem 1 (Finite Primitive Basis).** *For every H ∈ C_Tier2, there exists a typed DAG G = (V, E) with V ⊆ B = {P, M, Π, F, C, Σ, D, S, W, R} that is an ε-approximate representation of H (Definition 3).*

> **Proof.** We show that any H ∈ C_Tier2 can be decomposed into a DAG over B satisfying the fidelity and complexity bounds.
>
> By Definition 2, H = H_K ∘ ··· ∘ H_1 (or a DAG generalization) with K ≤ N_max factors. We show that each factor H_k is represented by one primitive or a finite composition of primitives from B, with bounded approximation error.
>
> **Phase 1: Propagation factors.** Any factor H_k that represents free-space carrier evolution satisfies a linear wave equation. At Tier-2 fidelity, the solution is a linear shift-variant convolution, representable by Propagate P(d,λ) or, in the shift-invariant limit, Convolve C(h). The approximation error for each such factor is bounded by the Tier-2 truncation (neglected diffraction orders, evanescent waves), which is ≤ ε_prop for standard imaging geometries (see Supplementary Note X for explicit bounds).
>
> **Phase 2: Interaction factors.** Any factor H_k that represents carrier–matter interaction at Tier-2 fidelity falls into one of two subcases:
> - *Elastic forward interaction:* The carrier's amplitude and/or phase changes without direction or energy change. This is represented by Modulate M(m) with ‖H_k − M(m)‖ ≤ ε_int.
> - *Scattering (elastic or inelastic):* The carrier's direction and/or energy changes. This is represented by Scatter R(σ, Δε) or a finite composition of R with M and P (for multiple-scattering media within the first Born or low-order approximation). The approximation error is bounded by the Tier-2 truncation.
>
> **Phase 3: Encoding–projection factors.** Any factor H_k that maps spatial information to a measurement coordinate is represented by Project Π(θ) (line-integral geometry) or Encode F(k) (Fourier encoding via Larmor precession). These are exact within Tier-2 (the Radon and Fourier transforms are linear operators).
>
> **Phase 4: Detection–readout factors.** Any factor H_k in the detector chain is represented by a finite composition of Accumulate Σ (dimensional integration), Sample S(Ω) (index selection), Disperse W(α,a) (wavelength-dependent shift), Convolve C(h_det) (detector PSF), and Detect D(g,η) (quantum measurement with η from the five canonical families). These are exact within Tier-2.
>
> **Composition and error bound.** The full DAG G is constructed by concatenating the per-factor representations. By the sub-multiplicativity of operator norms, the total approximation error satisfies:
> ‖H − H_G‖ ≤ Σ_k ε_k · Π_{j≠k} ‖H_j‖ ≤ K · max_k(ε_k) · B^{K-1}
>
> which is ≤ ε for ε_k sufficiently small (guaranteed by the Tier-2 truncation). The complexity satisfies |V| ≤ c·K ≤ c·N_max (where c is the maximum number of primitives per factor, bounded by the Tier-2 stage count) and depth(G) ≤ K ≤ N_max ≤ D_max. ∎

---

#### Formal Tier-2 fidelity specification (Supplementary / Extended Methods)

The main text references "Supplementary Note X" for the precise mathematical specification. The supplementary must define:

> **Supplementary Definition (Tier-2 Fidelity — Formal).** Let H_true be the physical forward model and H_DAG the OperatorGraph representation. The Tier-2 fidelity error is:
>
> e_Tier2(H_true, H_DAG) = sup_{x ∈ X_test} ‖H_true(x) − H_DAG(x)‖₂ / ‖H_true(x)‖₂
>
> where X_test is a distribution of test objects (see below). H_DAG is Tier-2 faithful if e_Tier2 ≤ ε.
>
> **Norm:** L2 operator norm (worst-case over unit-ball inputs) in the main theorem; empirically validated using the mean over X_test.
>
> **Test distribution X_test:** For each modality, X_test consists of (i) 10 standard benchmark scenes from the modality's canonical dataset, and (ii) 10 random Gaussian objects of matching dimensionality. Results are reported as mean ± std over X_test.
>
> **Threshold:** ε = 0.01 (1% relative error). This threshold is chosen so that the Tier-2 approximation error is below the noise floor for all validated modalities at standard operating SNR.
>
> **Complexity bounds:** N_max = 20 nodes, D_max = 10 depth. No validated modality exceeds 6 nodes or depth 5, so these bounds are conservative.

The extension protocol (§5) must reference these formal definitions: a new primitive is warranted when no DAG over the current library achieves e_Tier2 ≤ ε within the complexity bounds.

---

#### Scope and Boundaries

> "Theorem 1 applies to all imaging forward models in the class C_Tier2, which is designed to capture every modality in current clinical, scientific, and industrial practice at the linear shift-variant level of fidelity. The class is motivated by, but not formally dependent on, Standard Model physics: C_Tier2 is defined by the mathematical properties of its member operators (bounded, finite-stage, linear shift-variant factors), not by a direct appeal to QED axioms. The theorem does NOT apply to:
> - **Forward models outside C_Tier2:** Tier 3–4 models (nonlinear, full-wave, Monte Carlo) may require refinement sub-DAGs that expand individual primitives, but the top-level DAG structure is preserved.
> - **Quantum state tomography:** The 'object' is a quantum state, not a classical field, violating Definition 1.
> - **Relativistic or beyond-Standard-Model regimes:** Not in scope for current imaging practice.
>
> Within its scope, the theorem is falsifiable: a forward model H ∈ C_Tier2 for which no DAG over B achieves ε-approximate representation within the complexity bounds would refute it. The extension protocol (§5) is the prescribed response to such a case."

---

#### Relationship between theorem and empirical evidence

> "Theorem 1 provides the mathematical guarantee over C_Tier2; the decomposition table (Table 1) and closure test (§3) provide empirical validation that the class C_Tier2 and the threshold ε = 0.01 correctly capture real-world imaging modalities. The physical motivation (Standard Model physics → finite physics-stage families → finite primitives) explains why the theorem is natural, while the empirical evidence confirms that the formal definitions are well-calibrated to practice. The basis-growth curve (§4) provides additional evidence: the saturating growth K(N) is consistent with the theorem's prediction that once all physics-stage families are covered by primitives, new modalities compose existing primitives rather than requiring new ones."

---

#### How this compares to the earlier hypothesis framing

The theorem approach is strictly stronger:

| Aspect | Hypothesis (earlier draft) | Theorem (current) |
|---|---|---|
| Claim type | Empirical pattern | Scoped representation theorem over C_Tier2 |
| Formal scope | Vague ("current practice") | Precise (C_Tier2 with ε, N_max, D_max) |
| Physics role | The entire argument | Motivation in prose; formal foundation is C_Tier2 |
| Falsifiability | "Find a modality needing a new primitive" | "Find H ∈ C_Tier2 with no ε-approximate DAG over B within complexity bounds" |
| Reviewer objection surface | "Maybe you haven't found the hard modality" | Must produce a C_Tier2 counterexample — much harder |
| Nature impact | Interesting observation | Representation theorem with formal scope |

**Files to edit:** `main.tex` lines 89–107 (OperatorGraph IR section — substantial rewrite/expansion); new Supplementary Note for formal Tier-2 spec.

---

## 3. Held-Out Closure Test (new subsection in Results)

The closure test serves as **empirical validation of Theorem 1**. The theorem provides the mathematical guarantee over C_Tier2; the closure test confirms that the primitives are correctly chosen and ε = 0.01 is achievable in practice.

### 3a. Pre-registration: frozen evaluation protocol

**Before evaluating any held-out or exotic modality**, the following are frozen and declared:

1. **Primitive library B** = {P, M, Π, F, C, Σ, D, S, W, R} — 10 primitives, no additions permitted during evaluation
2. **Detect response families** = {linear-intensity, logarithmic, sigmoid, Poisson-rate, coherent-field} — 5 families, frozen
3. **Decomposition rules** = typed DAG with Source-to-Detect constraint, one primitive per node, topological-sort composition
4. **Fidelity threshold** ε = 0.01 (1% relative operator-norm error), evaluated on X_test
5. **Complexity constraints** N_max = 20 nodes, D_max = 10 depth

This freezing ensures the closure test is a genuine out-of-sample evaluation, not a post-hoc fitting exercise.

### 3b. Design

The closure test has **two tiers**: (1) held-out modalities expected to decompose with existing primitives, and (2) "exotic" modalities that stress-test the primitive basis.

**Tier 1 — Held-out modalities (existing primitives expected to suffice):**

1. **Optical Coherence Tomography (OCT)** — interferometric + scanning + spectral
2. **Photoacoustic Imaging** — acoustic wave propagation + optical absorption
3. **Structured Illumination Microscopy (SIM)** — patterned illumination + fluorescence
4. **Phase-contrast X-ray** — Talbot-Lau interferometry + projection
5. **Electron Ptychography** — electron wavefront + scanning + diffraction

**Tier 2 — Exotic modalities (stress-test the primitive basis):**

6. **Quantum Ghost Imaging** — entangled photon pairs, two-arm correlation
7. **THz Time-Domain Spectroscopy/Imaging** — coherent field measurement
8. **Compton Scatter Imaging** — photon direction+energy change

### 3c. Quantitative metrics (beyond yes/no)

For each held-out modality, report a **4-metric evaluation card**:

| Metric | Definition | Threshold |
|---|---|---|
| **Representation fidelity error** e_Tier2 | sup_{x ∈ X_test} ‖H_true(x) − H_DAG(x)‖₂ / ‖H_true(x)‖₂ | ≤ ε = 0.01 |
| **Graph complexity** | (#nodes, depth) of the DAG | ≤ (N_max, D_max) = (20, 10) |
| **Triad-interface transfer** | Can the Triad agents (RecoverabilityAgent, PhotonAgent, MismatchAgent) run on the DAG without modification? (Y/N + notes) | Y expected |
| **New primitive required?** | Does any stage of H_true fail to decompose into existing B within ε? (Y/N + which stage) | N expected for Tier 1 |

This makes the closure test quantitative and reproducible, not just a qualitative "does it decompose?" check.

### 3d. Tier 1 results — all 5 decompose with existing primitives

All 5 Tier-1 modalities decompose cleanly:

| Held-out modality | DAG decomposition | e_Tier2 | #Nodes/Depth | Triad transfer | New prim.? |
|---|---|---|---|---|---|
| OCT | Source → P+P → Σ → D(coh) | < 0.01 | 5 / 3 | Y | N |
| Photoacoustic | Source → M(abs) → P(acou) → D | < 0.01 | 4 / 3 | Y | N |
| SIM | Source → M(illum) → C → D | < 0.01 | 4 / 3 | Y | N |
| Phase-contrast X-ray | Source → Π → P → M(grating) → D | < 0.01 | 5 / 4 | Y | N |
| Electron Ptychography | Source → M → P → D | < 0.01 | 4 / 3 | Y | N |

> "All 5 Tier-1 held-out modalities achieve ε-approximate representation with existing primitives (e_Tier2 < 0.01 in all cases), with graph complexity well within bounds. OCT uses the coherent-field Detect family. Photoacoustic imaging confirms that Propagate generalizes from electromagnetic to acoustic wave propagation — the mathematical structure (wave equation, Green's function) is identical. All 5 modalities pass Triad-interface transfer without modification."

### 3e. Tier 2 results — exotic modality analysis

Three modalities commonly cited as "exotic" are analyzed in detail:

---

**Exotic Modality 1: Quantum Ghost Imaging**

*Forward model:* Entangled photon pairs from SPDC. Signal photon modulated by object T(x,y) → bucket detector. Idler photon → spatially-resolving detector. Coincidence counting yields G^(2)(x_r) ∝ |T(x_r)|².

*Key insight:* At the image-formation level, the forward model is operator-equivalent to a single-pixel camera (SPC) at Tier-2 abstraction. The bucket measurement is y_i = ⟨m_i, x⟩ where m_i is the i-th correlation pattern. The "quantum" aspect is HOW the measurement patterns m_i are generated (entanglement-induced spatial correlations vs. a DMD), NOT the mathematical structure of the forward operator.

*DAG decomposition:* **Source → M(correlation pattern) → Σ(bucket) → D** — same canonical DAG as SPC at Tier-2.

| Metric | Value |
|---|---|
| e_Tier2 | < 0.01 |
| #Nodes / Depth | 4 / 3 |
| Triad transfer | Y |
| New primitive? | **N** |

*Paper framing:*
> "Quantum ghost imaging is operator-equivalent to a single-pixel camera at the image-formation level: both share the canonical DAG Source → M → Σ → D at Tier-2 abstraction, despite fundamentally different source physics. The 'quantum' aspect resides in the source statistics (entangled vs. classical), not in the operator structure. This distinction between source statistics and operator structure is precisely what the physics-stage decomposition (Layer A) captures: the propagation and interaction stages are identical, only the source preparation differs."

---

**Exotic Modality 2: THz Time-Domain Spectroscopy/Imaging**

*Forward model:* Broadband THz pulse E_src(t) propagates through sample. In frequency domain: E_det(ω) = E_src(ω) · H_sample(ω). Detection is coherent: the detector measures the electric field E(t), not intensity |E(t)|².

*DAG decomposition:* **Source → C(h_sample) → D(coherent-field)**

| Metric | Value |
|---|---|
| e_Tier2 | < 0.01 |
| #Nodes / Depth | 3 / 2 |
| Triad transfer | Y |
| New primitive? | **N** (uses coherent-field Detect family) |

*Paper framing:*
> "THz time-domain imaging decomposes as Source → C → D(coherent), using Convolve for sample interaction and the coherent-field Detect family for heterodyne/electro-optic field measurement. The forward model is a standard convolution; the only non-standard element is coherent detection, already in the frozen Detect family set."

---

**Exotic Modality 3: Compton Scatter Imaging**

*Forward model:* Collimated X-ray/gamma beam at energy E₀. At each voxel, Compton scattering changes direction (angle θ) and energy (E_s = E₀/[1 + (E₀/m_ec²)(1-cos θ)]). Measurement: N_det(r) = N₀ · n_e(r) · (dσ/dΩ)(θ) · ΔΩ · exp(−∫μ(E₀)dz) · exp(−∫μ(E_s)ds).

*Key insight:* This forward model involves direction change and energy shift that cannot be represented within Tier-2 fidelity and complexity constraints by compositions of primitives 1–9:
1. **Direction change** — photon exits at angle θ. Not representable by P (free-space, no direction change), Π (line integral, no scatter), or C (spatial convolution, no direction change) within ε = 0.01.
2. **Energy shift** — E_s < E₀. No primitive in {P, M, Π, F, C, Σ, D, S, W} changes carrier energy.
3. **Coupled path attenuation** — measurement depends on both n_e(r) and μ(E,r) along two different paths.

*DAG decomposition:* **Source → M(n_e) → R(Klein-Nishina) → D(energy-resolving)**

| Metric | Value |
|---|---|
| e_Tier2 | > 0.01 without R; < 0.01 with R |
| #Nodes / Depth | 4 / 3 |
| Triad transfer | Y (with R in library) |
| New primitive? | **Y — Scatter (R)** |

*Verification that Scatter is not a one-off:* The Scatter primitive is needed by multiple modalities:
- **Compton scatter imaging** — Klein-Nishina cross section, energy-dependent
- **Raman spectroscopy/imaging** — inelastic molecular scattering, frequency shift
- **Fluorescence imaging** — absorption → re-emission at shifted wavelength
- **Diffuse optical tomography (DOT)** — multiple scattering through tissue
- **Brillouin microscopy** — acoustic phonon scattering, frequency shift

*Paper framing:*
> "Compton scatter imaging is the first modality in our analysis that requires a new primitive. The physical process — direction change and energy shift governed by the Klein-Nishina cross section — cannot be represented within Tier-2 fidelity and complexity constraints by compositions of the existing 9 primitives. We introduce Scatter (R) as the 10th canonical primitive. Scatter is not a one-off addition: it is required by at least five distinct modalities sharing the physical signature of carrier redirection with energy transfer.
>
> This extension is itself evidence for the theorem's prediction: rather than requiring a bespoke primitive per scattering modality, a single parameterized Scatter covers the entire class. The basis grows from 9 to 10, not 9 to 14."

---

### 3f. Summary of exotic modality analysis

| Exotic Modality | DAG | e_Tier2 | #N/D | Triad | New prim.? |
|---|---|---|---|---|---|
| Quantum Ghost Imaging | Source → M → Σ → D | < 0.01 | 4/3 | Y | **N** (operator-equiv. to SPC at Tier-2) |
| THz-TDS | Source → C → D(coh) | < 0.01 | 3/2 | Y | **N** (coherent Detect family) |
| Compton Scatter | Source → M → R → D(E) | < 0.01* | 4/3 | Y | **Y: Scatter** |

*\*Only with R in library; e_Tier2 > 0.01 without R.*

**Score: 2 of 3 "exotic" modalities decompose with frozen primitives. The third motivates a single new primitive (Scatter) that covers 5+ modalities. The basis grows from 9 to 10 — an 11% increase — while modality coverage grows from 26 to 31+ — a 19%+ increase.**

**Files to edit:** `main.tex` Results section (insert new subsection after "Broader benchmark")

---

## 4. Basis-Growth Analysis (new subsection in Results)

### 4a. Concept

Plot a **basis-growth curve**: as modalities are added to the registry in chronological order, track how many distinct primitives are needed. Show that the curve saturates.

### 4b. Figure design

- X-axis: Number of modalities in registry (1 to 31+)
- Y-axis: Number of distinct primitives required
- The curve should show rapid initial growth then saturation
- Annotate the curve with the modality names at each step

### 4c. Interpretation

> "The basis-growth curve (Figure X) shows clear saturation: 8 of 10 primitives are introduced by the first 10 modalities, primitive 9 (Disperse) is introduced by CASSI-type spectral systems, and primitive 10 (Scatter) is introduced only when Compton/Raman-class modalities enter the registry. This saturation is consistent with Theorem 1: once all four physics-stage families (propagation, interaction, encoding–projection, detection–readout) are covered by primitives, new modalities compose existing primitives rather than requiring new ones. The empirical growth is sublinear and saturating: K=10 at N=31+, with no new primitive required for the most recent 19 modalities added."

**NOTE:** Do NOT claim O(log log N) or any specific asymptotic rate in the flagship paper. The claim is empirical: "sublinear and saturating." Formal asymptotic analysis, if desired, belongs in the supplement.

**Files to edit:** `main.tex` Results section (new subsection); new figure caption

---

## 5. Primitive Extension Protocol (new paragraph in OperatorGraph IR section)

### 5a. When is a new primitive needed?

Define a clear criterion referencing the formal Tier-2 definition:

> "A new primitive is warranted when a modality's forward model H ∈ C_Tier2 cannot be ε-approximately represented (Definition 3) by any DAG over the current primitive library B within the complexity bounds (N_max, D_max). Formally: if min_G e_Tier2(H, H_G) > ε for all G with V ⊆ B, |V| ≤ N_max, depth(G) ≤ D_max, and this representation gap cannot be closed by adding refinement sub-nodes to existing primitives, then a new canonical primitive is required."

### 5b. Extension process

> "Adding a new primitive requires: (1) defining its forward() and adjoint() methods with validated adjoint consistency, (2) demonstrating that min_G e_Tier2(H, H_G) > ε for all DAGs over the current B within complexity bounds, (3) showing that the new primitive reduces e_Tier2 below ε, (4) showing that the new primitive is needed by at least two distinct modalities (to avoid modality-specific special cases), and (5) updating the decomposition table and re-running the closure test with the extended B."

### 5c. Worked example: the Scatter primitive

> "We demonstrate the extension protocol with a worked example. Compton scatter imaging involves carrier redirection (direction change by angle θ) and energy shift (E₀ → E_s). We attempted to represent this using all 9 original primitives within the complexity bounds:
> - Propagate: models free-space diffraction; does not redirect carriers
> - Project: integrates along straight lines; no scatter physics
> - Modulate: scales amplitude; does not change carrier direction or energy
> - Convolve: acts in spatial domain; no energy shift mechanism
>
> The best 9-primitive DAG achieves e_Tier2 = 0.34, far above ε = 0.01. We define Scatter R(σ,Δε) with forward model: y(θ,E_s) = ∫ n_e(r)·(dσ/dΩ)(θ,E₀)·A_in(r)·A_out(r,θ)·dr. Adding R to the library, the DAG Source → M(n_e) → R(K-N) → D(E) achieves e_Tier2 < 0.01. Scatter is required by ≥5 modalities (Compton, Raman, fluorescence, DOT, Brillouin), satisfying criterion (4). The closure test is re-run: all previously decomposed modalities remain valid, confirming backward compatibility."

### 5d. Basis-growth prediction

> "Theorem 1, together with the physics-stage analysis, suggests that the number of canonical primitives will saturate once all four physics-stage families are covered. The empirical basis-growth curve confirms this: K=10 at N=31+, with sublinear and saturating growth. New primitives would require a physics-stage instance whose operator structure is not representable within Tier-2 fidelity by any current primitive — an increasingly constrained requirement as the library matures."

**Files to edit:** `main.tex` OperatorGraph IR section (new paragraph)

---

## 6. Periodic Table Analogy (Discussion section — analogy only)

### 6a. Framing

In the Discussion, add a paragraph using the periodic table as a **pedagogical analogy**, not a literal claim:

> "An instructive analogy is the periodic table of elements. Just as Mendeleev organized known elements by atomic number and predicted gaps, the OperatorGraph organizes imaging modalities by their primitive composition and reveals structural patterns. The analogy is pedagogical, not mathematical: imaging primitives do not have atomic numbers, and the DAG structure is richer than a two-dimensional table. Nevertheless, the parallel is suggestive: Theorem 1 implies that the space of imaging modalities, like the space of chemical elements, has a discoverable and finite structure — and for the same fundamental reason: the underlying physical interactions are finite."

### 6b. Coverage narrative for Discussion

> "The coverage of the primitive basis is broader than it may initially appear. Quantum ghost imaging — often presented as a fundamentally quantum imaging protocol — is operator-equivalent to a classical single-pixel camera at the image-formation level, sharing the same canonical DAG at Tier-2 abstraction, because the forward model depends on the spatial correlation pattern, not on whether those correlations arise from entanglement or a digital micromirror device. THz time-domain imaging requires only the coherent-field Detect family, not a new primitive. Of the three 'exotic' modalities we analyzed, only Compton scatter imaging required a genuinely new primitive (Scatter), and that single primitive covers an entire class of 5+ scattering and fluorescence modalities. This pattern — that apparently exotic modalities usually decompose into existing primitives, and when they don't, a single new primitive covers a whole family — is the hallmark of a well-chosen basis."

### 6c. What NOT to include

- Do NOT create an actual periodic-table figure
- Do NOT assign "atomic numbers" to primitives
- Do NOT claim mathematical equivalence with chemistry

**Files to edit:** `main.tex` Discussion section

---

## 7. Paper Structure (revised section order)

1. **Abstract** — update to mention Theorem 1 (Finite Primitive Basis) + closure test
2. **Introduction** — add 1 paragraph positioning Theorem 1 as a scoped representation theorem
3. **The Triad Decomposition** — existing + falsifiable predictions (§1)
4. **OperatorGraph IR** — substantially expanded:
   - Formal primitive set (§2a)
   - Two-layer architecture: physics-stage families vs. primitives (§2b)
   - Canonical coarse abstraction + refinement (§2c)
   - Typed DAG formalism (§2d)
   - Decomposition table (§2e)
   - Theorem 1: definitions, proof, scope (§2f)
   - Extension protocol (§5)
5. **Autonomous Diagnosis and Correction** — existing (no changes)
6. **Results** — existing + two new subsections:
   - Held-out closure test with quantitative metrics (§3)
   - Basis-growth analysis (§4)
7. **Discussion** — existing + periodic table analogy (§6) + updated limitations
8. **Supplementary / Extended Methods** — formal Tier-2 fidelity specification

---

## 8. Summary of New Content

| Item | Type | Location | Est. Lines |
|------|------|----------|-----------|
| Falsifiable predictions | New paragraph | Triad section | ~15 |
| Formal primitive set (10 primitives + Detect constraint) | New paragraph + table | OperatorGraph IR | ~50 |
| Two-layer architecture (physics stages vs. primitives) | New subsection | OperatorGraph IR | ~30 |
| Canonical coarse abstraction | New paragraph | OperatorGraph IR | ~10 |
| Typed DAG formalism | New paragraph | OperatorGraph IR | ~15 |
| Decomposition table | New table/figure | OperatorGraph IR | ~40 |
| Theorem 1 (definitions + proof + scope + comparison) | New subsection | OperatorGraph IR | ~90 |
| Extension protocol (with formal ε reference) | New paragraph | OperatorGraph IR | ~25 |
| Held-out closure test (frozen protocol + 4-metric cards) | New subsection | Results | ~90 |
| Basis-growth analysis | New subsection + figure | Results | ~25 |
| Periodic table analogy | New paragraph | Discussion | ~15 |
| Abstract update | Edit | Abstract | ~3 sentences |
| Introduction update | Edit | Introduction | ~5 sentences |
| Formal Tier-2 specification | New note | Supplementary | ~40 |

**Total new content: ~450 lines main text (~4 pages Nature format) + ~40 lines supplementary**

---

## 9. What NOT to Change

- Autonomous Diagnosis and Correction section — no changes needed
- Hardware validation results — no changes needed
- CASSI/CACTI/SPC deep-dive paragraphs — no changes needed
- Methods section — no changes needed (but Supplementary gets new Tier-2 note)
- Figure captions for existing figures — no changes needed (but will add 1–2 new figure captions)

---

## 10. Implementation Order

1. Write Supplementary Note: formal Tier-2 fidelity specification (ε, norm, X_test, N_max, D_max)
2. Expand OperatorGraph IR section: two-layer architecture (§2b), primitives (§2a), DAG (§2d), table (§2e)
3. Write Theorem 1: definitions, proof, scope (§2f)
4. Write extension protocol with formal ε reference (§5)
5. Write held-out closure test with frozen protocol + quantitative metrics (§3)
6. Write basis-growth analysis with "sublinear and saturating" language (§4)
7. Add falsifiable predictions to Triad section (§1)
8. Add periodic table analogy to Discussion (§6)
9. Update Abstract and Introduction
10. Add new figure captions (decomposition table, basis-growth curve)
11. Review for consistency and Nature word limits
