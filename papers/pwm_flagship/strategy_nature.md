# PWM Nature Paper Strategy

**Paper:** "Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging"
**Target:** Nature
**Date:** 2026-02-27

---

## Table of Contents

1. [Current Paper Status](#1-current-paper-status)
2. [Critical Gaps a Nature Reviewer Will Flag](#2-critical-gaps-a-nature-reviewer-will-flag)
3. [5 Modalities to Add (Prioritized)](#3-5-modalities-to-add-prioritized)
4. [What NOT to Add](#4-what-not-to-add)
5. [Collaborator Strategy](#5-collaborator-strategy)
6. [Before vs. After Comparison](#6-before-vs-after-comparison)
7. [Implementation Priority & Effort Estimates](#7-implementation-priority--effort-estimates)
8. [Paper Structure Improvements](#8-paper-structure-improvements)

---

## 1. Current Paper Status

### Authors
1. **Chengshuai Yang** (corresponding) — NextGen PlatformAI C Corp, USA
2. **Xin Yuan** — School of Engineering, Westlake University, Hangzhou, China
3. ~~David J. Brady~~ (commented out, pending invitation)
4. ~~Steve B. Jiang~~ (commented out, pending invitation)

### Validated Modalities (7, 4-Scenario Protocol)

| Modality | Carrier | Oracle Gain | Hardware Validated |
|---|---|---|---|
| CASSI | Photon (incoherent) | +0.76/+6.50 dB | Yes (TSA real data) |
| CACTI | Photon (incoherent) | +10.21 dB | Yes (EfficientSCI real data) |
| SPC | Photon (incoherent) | +7.71/+10.38 dB | No |
| Lensless | Photon (incoherent) | +3.55 dB | No |
| Ptychography | Electron (coherent) | +7.09 dB | Yes (4D-STEM SrTiO₃) |
| CT | X-ray photon | +10.68 dB | Yes (FIPS walnut, HTC sinograms) |
| MRI | Nuclear spin | +1.75 to +7.14 dB | Yes (M4Raw multi-coil) |

### Held-out Closure Test (8 modalities, all pass)
OCT, Photoacoustic, SIM, Phase-contrast X-ray, Electron Ptychography, Ghost Imaging, THz-TDS, Compton Scatter

### Claimed Carrier Coverage
5 families: Photons, Electrons, Spins, Acoustic waves, Particles

### Key Strengths
- Finite Primitive Basis Theorem (11 primitives, sufficient and minimal)
- Triad Decomposition (3 gates, Gate 3 dominates in 9/9 configs)
- +0.8 to +10.7 dB correction via forward-model calibration alone
- Hardware validation on 5 real instruments
- Zero-shot cross-carrier generalization (<0.5 dB transfer gap)

---

## 2. Critical Gaps a Nature Reviewer Will Flag

### Gap 1: No acoustic carrier validation (CRITICAL)

The paper claims 5 carrier families but validates only 4. Acoustic is completely missing — not even in the held-out closure test. A Nature reviewer will immediately write:

> *"The authors claim universality across 5 carrier families but present zero validation for acoustic imaging. This significantly weakens the universality claim."*

**Fix:** Add Ultrasound.

### Gap 2: No biology modality (MAJOR)

Nature's core readership is biologists and biomedical scientists. The paper has zero biology-relevant modalities. CT and MRI are clinical/radiology, not biology per se. A Nature reviewer (especially one from the biology side) will ask:

> *"How does this framework apply to the imaging modalities that dominate modern biology — microscopy and cryo-EM?"*

**Fix:** Add Cryo-EM and/or Fluorescence Microscopy.

### Gap 3: Optical photons are overrepresented (MODERATE)

4 of 7 validated modalities are incoherent optical photon systems (CASSI, CACTI, SPC, Lensless). This makes the paper look like a snapshot compressive imaging paper with a few other modalities added, rather than a truly universal framework.

**Fix:** Adding non-optical modalities (Ultrasound, Cryo-EM, CBCT) rebalances the portfolio.

### Gap 4: No coherent 3D optical imaging (MODERATE)

All optical modalities are 2D incoherent. The paper doesn't demonstrate that the primitives handle coherent 3D optical reconstruction (holography).

**Fix:** Add Compressive Holography.

### Gap 5: CT is only parallel-beam (MINOR)

The CT validation uses parallel-beam geometry (academic micro-CT). Clinically, cone-beam CT dominates. Adding CBCT shows the framework handles the clinically relevant geometry.

**Fix:** Add CBCT.

---

## 3. 5 Modalities to Add (Prioritized)

### #1. Ultrasound — MUST ADD

**Gap addressed:** Acoustic carrier (Gap 1 — the most critical)

**Why non-negotiable:**
- Fills the only missing carrier family in the validation
- Clinically ubiquitous (~200,000 scanners worldwide)
- Every Nature reader has had an ultrasound
- Speed-of-sound mismatch is a textbook Gate 3 problem

**OperatorGraph DAG:**
```
Source(x) → C(h_PSF) → S(Ω_scanlines) → D(g, η_linear) → y
```

| Node | Primitive | Physical action |
|------|-----------|----------------|
| `C(h_PSF)` | Convolve | Spatially-varying acoustic PSF (diffraction + aberration from tissue heterogeneity) |
| `S(Ω)` | Sample | Scan-line sampling pattern (linear, sector, phased array) |
| `D(g, η)` | Detect | Piezoelectric transducer array (linear field response) |

**Canonical chain:** `C → S → D` (3 nodes, depth 3)

**Gate 3 mismatch parameters:**

| Parameter | Primitive | Description | Typical Error |
|-----------|-----------|-------------|---------------|
| **Speed of sound** | **C** | **Assumed 1540 m/s vs. actual tissue variation (1450–1600 m/s)** | **3–7%** |
| Attenuation model | C | Frequency-dependent absorption (0.5 dB/cm/MHz assumed, varies by tissue) | 20–50% |
| Element sensitivity | D | Per-element gain variation in transducer array | 2–10% |
| Beamforming geometry | S | Assumed vs. actual element positions | 0.1–0.5 mm |
| Phase aberration | C | Tissue-induced wavefront distortion | 10–50 ns |

**Testable prediction:** Speed-of-sound mismatch of 3% should cause ~2 mm geometric distortion and +2–4 dB correction gain through phase-aberration correction. Gate 3 should dominate over Gate 2 (thermal noise) at standard clinical imaging depths (< 15 cm).

**Datasets:** PICMUS (Plane-wave Imaging Challenge in Medical Ultrasound), simulation + phantom data. Also: IUS open datasets.

**Effort:** Low. All primitives exist. Simple forward model. Public datasets available.

---

### #2. Cryo-EM — MUST ADD

**Gap addressed:** Biology audience (Gap 2 — Nature's core readership)

**Why critical:**
- **2017 Nobel Prize in Chemistry** (Dubochet, Frank, Henderson) — instant recognition
- Biology/structural biology is Nature's largest readership segment
- CTF estimation is the rate-limiting calibration step — **textbook Gate 3**
- A 50nm defocus error at 300kV limits resolution from 2Å to 3Å
- ~$5M per instrument, ~2,000 cryo-EMs worldwide, used for drug discovery

**OperatorGraph DAG:**
```
Source(x_3D) → P(e⁻ wave, λ) → M(specimen) → P(objective lens) → C(CTF) → D(DED) → micrograph
```

| Node | Primitive | Physical action |
|------|-----------|----------------|
| `P(λ)` | Propagate | Electron plane wave illumination (300 kV, λ ≈ 0.02 Å) |
| `M(specimen)` | Modulate | Specimen potential modulates electron phase + amplitude |
| `P(obj)` | Propagate | Objective lens transfer (spherical aberration Cs) |
| `C(CTF)` | Convolve | Contrast Transfer Function (oscillatory, defocus-dependent) |
| `D(DED)` | Detect | Direct electron detector (counting mode, DQE > 0.5) |

**Canonical chain:** `P → M → P → C → D` (5 nodes, depth 5)

**Gate 3 mismatch parameters:**

| Parameter | Primitive | Description | Typical Error |
|-----------|-----------|-------------|---------------|
| **Defocus Δf** | **C(CTF)** | **CTF zero-crossings shift with defocus** | **50–500 nm** |
| Spherical aberration Cs | P(obj) | Lens aberration coefficient | 0.1–2% |
| Astigmatism | C(CTF) | Elliptical CTF from asymmetric defocus | 10–100 nm |
| Beam tilt | P(λ) | Illumination axis misalignment | 0.1–1 mrad |
| Magnification anisotropy | M | Distortion in specimen plane | 0.1–1% |
| Ice thickness variation | M | Sample embedding quality | 20–100 nm |

**Testable prediction:** Defocus mismatch of 200nm should degrade reconstruction resolution by >0.5Å. PWM CTF correction should recover +3–6 dB in a single-particle reconstruction. The resolution ceiling (FSC=0.143) should shift toward higher spatial frequency after Gate 3 correction.

**Datasets:** EMPIAR (Electron Microscopy Public Image Archive) — thousands of public datasets. RELION/cryoSPARC benchmark datasets.

**Effort:** Medium. Shares `P` (Propagate) with ptychography. CTF convolution is well-understood. The main work is implementing the CTF model and connecting to EMPIAR data.

---

### #3. CBCT (Cone-Beam CT) — STRONGLY RECOMMENDED

**Gap addressed:** Clinical depth (Gap 5) + hooks Jiang

**Why important:**
- Extends CT from academic parallel-beam to clinically dominant cone-beam geometry
- **Critical for attracting Prof. Steve Jiang** (his deepest research area, 50+ papers)
- Scatter correction is CBCT's #1 unsolved problem — a perfect Gate 3 demonstration
- ~10,000 CBCT-equipped linear accelerators in the US (radiation oncology workhorse)
- Beam hardening tests the `Λ` (Transform) primitive with real clinical physics

**OperatorGraph DAG:**
```
Source(x_3D) → Λ(Beer-Lambert) → Π(θ, cone) → C(scatter PSF) → D(flat-panel) → y
```

| Node | Primitive | Physical action |
|------|-----------|----------------|
| `Λ(BL)` | Transform | Beer-Lambert exponential attenuation (polychromatic beam hardening) |
| `Π(θ, cone)` | Project | Cone-beam projection geometry (divergent rays from point source) |
| `C(scatter)` | Convolve | Scatter contamination (patient-dependent kernel) |
| `D(fp)` | Detect | Flat-panel detector (gain nonuniformity, lag) |

**Canonical chain:** `Λ → Π → C → D` (4 nodes, depth 4)

**Gate 3 mismatch parameters:**

| Parameter | Primitive | Description | Typical Error |
|-----------|-----------|-------------|---------------|
| **Scatter kernel** | **C** | **Patient-dependent scatter distribution, never perfectly known** | **10–30% of primary signal** |
| Geometric calibration | Π | Source-detector alignment drift | 0.1–0.5 mm |
| Beam hardening | Λ | Polychromatic spectrum vs. monoenergetic assumption | 5–15% HU error |
| Detector gain | D | Pixel-to-pixel flat-panel response variation | 1–5% |
| Detector lag | D | Temporal ghosting from previous projections | 1–3% |

**Testable prediction:** Scatter kernel mismatch should produce cupping artifacts and +3–6 dB correction gain. Geometric calibration drift of 1mm should cause +2–4 dB degradation with ~100% oracle recovery (scalar CoR parameter, clean minimum).

**Datasets:** CatPhan phantom simulation datasets; AAPM Grand Challenge data; synthetic CBCT from CT volumes.

**Effort:** Low. Extends the validated parallel-beam CT template. The `Π` (Project) primitive already exists; cone-beam is a parameter change. Scatter adds a `C` (Convolve) node.

---

### #4. Compressive Holography — STRONGLY RECOMMENDED

**Gap addressed:** Coherent 3D optical imaging (Gap 4) + hooks Brady

**Why important:**
- **Critical for attracting Prof. David Brady** (he won the 2023 Optica Leith Medal for inventing this)
- Demonstrates PWM handles coherent 3D reconstruction — fundamentally different from incoherent 2D (CASSI/CACTI/SPC)
- The forward model is elegant and compact: `P → M → P → D`
- 3D volumetric reconstruction from a single 2D hologram is a dramatic demonstration
- Compressive holography is the inverse problem that proved sparsity-based reconstruction works in optics

**OperatorGraph DAG:**
```
Source(x_3D(r,z)) → P(d₁,λ) → M(object) → P(d₂,λ) → D(|·|², η_sq) → hologram
```

| Node | Primitive | Physical action |
|------|-----------|----------------|
| `P(d₁,λ)` | Propagate | Reference beam free-space propagation |
| `M(object)` | Modulate | Object amplitude + phase modulation |
| `P(d₂,λ)` | Propagate | Object-to-detector propagation (Fresnel) |
| `D(\|·\|²)` | Detect | Intensity square-law (hologram = interference pattern) |

**Canonical chain:** `P → M → P → D` (4 nodes, depth 4)

**Gate 3 mismatch parameters:**

| Parameter | Primitive | Description | Typical Error |
|-----------|-----------|-------------|---------------|
| **Propagation distance** | **P** | **Recording distance error (defocus)** | **0.1–1% of z** |
| Reference beam angle | P | Off-axis angle calibration | 0.01–0.1 deg |
| Wavelength | P | Laser wavelength drift or uncertainty | 0.01–0.1 nm |
| Twin-image contamination | D | Incomplete twin-image suppression (on-axis holography) | Structural |
| Pixel pitch | D | Detector pixel size calibration | 0.1–0.5% |

**Testable prediction:** Propagation distance mismatch of 1% should produce defocus artifacts and +3–8 dB correction gain. Twin-image suppression via correct forward model should outperform algorithmic twin-image removal. Gate 3 should dominate for in-focus objects.

**Datasets:** Public digital holography datasets (DHM, off-axis holography benchmarks). Simulated Fresnel holograms from 3D phantom objects.

**Effort:** Medium. Coherent propagation shares Fresnel/angular-spectrum code with ptychography. The `P` primitive is already implemented.

---

### #5. Fluorescence Microscopy — RECOMMENDED

**Gap addressed:** Deepens biology coverage (Gap 2) + tests `Λ` (Transform)

**Why important:**
- Every biology lab has a fluorescence microscope — universal relevance for Nature's biology readers
- PSF estimation / deconvolution is the daily calibration challenge — pure Gate 3
- Super-resolution variants (STED, PALM, STORM) won the **2014 Nobel Prize in Chemistry**
- Bridges computational imaging to the massive life sciences tools market
- Tests the `Λ` (Transform) primitive for the fluorescence emission nonlinearity

**OperatorGraph DAG:**
```
Source(x_fluorophores) → M(I_excitation) → Λ(fluorescence) → C(h_emission) → D(camera) → y
```

| Node | Primitive | Physical action |
|------|-----------|----------------|
| `M(I_exc)` | Modulate | Excitation illumination pattern modulates fluorophore activation |
| `Λ(fluor)` | Transform | Stokes-shifted fluorescence emission (nonlinear: absorption → re-emission at longer λ) |
| `C(h_em)` | Convolve | Emission PSF (diffraction-limited Airy disk + aberrations) |
| `D(cam)` | Detect | sCMOS/EMCCD camera (Poisson photon statistics) |

**Canonical chain:** `M → Λ → C → D` (4 nodes, depth 4)

**Gate 3 mismatch parameters:**

| Parameter | Primitive | Description | Typical Error |
|-----------|-----------|-------------|---------------|
| **Emission PSF** | **C** | **Theoretical Airy disk vs. actual aberrated PSF** | **5–20% Strehl ratio** |
| Refractive index mismatch | C | Coverslip/immersion oil/sample RI variations | 0.01–0.05 RI units |
| Excitation uniformity | M | Non-uniform illumination field | 5–15% variation |
| Photobleaching | Λ | Time-dependent fluorophore deactivation | Cumulative |
| Chromatic aberration | C | Excitation vs. emission wavelength focus shift | 50–200 nm axial |
| Spherical aberration (depth) | C | PSF degrades with imaging depth in tissue | Depth-dependent |

**Testable prediction:** PSF mismatch (theoretical Airy disk vs. measured PSF) should limit deconvolution resolution. PWM correction using measured/estimated PSF should recover +2–5 dB. Refractive index mismatch at 50μm depth should produce >3 dB degradation.

**Datasets:** ISBI deconvolution challenge datasets; Hagen et al. PSF estimation benchmarks; BioImage Model Zoo test data.

**Effort:** Medium. All primitives exist. PSF convolution model is standard. Public datasets available from deconvolution challenges.

---

## 4. What NOT to Add

Save these for post-publication or supplementary material:

| Modality | Why not now | Better use |
|----------|-----------|------------|
| **Smartphone photography** | Not a single modality — it's 8 features, each a separate DAG. Adding all dilutes focus. | Discussion paragraph: "the same 11 primitives describe a phone camera" + supplementary note with DAGs |
| **PET** | Medium effort. Less unique than ultrasound for filling carrier gaps. Jiang's involvement is better served by CBCT first. | Post-acceptance extension with Jiang |
| **Gigapixel camera** | Less fundamental than holography. Doesn't add a new carrier or regime. | Post-publication, for Brady's deeper involvement |
| **4D-MRI** | Variant of existing MRI. Adds temporal dimension but no new carrier or primitive. | Supplementary note showing temporal extension |
| **SAR** | Interesting but niche audience for Nature. | Future paper on defense/remote sensing applications |
| **JWST** | Would be spectacular but requires significant domain knowledge and effort. | Discussion paragraph: "the same framework applies to telescope wavefront sensing" |
| **LIGO** | Fascinating but too far from imaging. | One-sentence mention in "beyond imaging" discussion |

---

## 5. Collaborator Strategy

### Prof. Xin Yuan (Westlake University) — ALREADY ADDED

**Status:** Co-author #2 in `main.tex`. Contribution package ready at `contribution_packages/contribution_yuan.md`.

**Role in paper:**
- Validates GAP-TV / EfficientSCI parameters
- Confirms CASSI/CACTI forward model specs and 5-parameter mismatch model
- Reviews real-data experimental protocols (TSA + EfficientSCI)
- Manuscript review of empirical validation sections

**Key connection:** Central figure in snapshot compressive imaging. Developed GAP-TV (primary solver) and EfficientSCI (CACTI validation). Bridges to Brady via their Duke collaboration.

**No action needed.**

---

### Prof. David Brady (University of Arizona) — INVITE WITH HOLOGRAPHY

**Profile:** Goodman Endowed Chair, Wyant College of Optical Sciences. Optica/IEEE/SPIE Fellow. ~17,600 citations. Invented CASSI. Won 2023 Leith Medal for compressive holography. Nature 2012 paper on gigapixel cameras.

**Current hook:** CASSI is already validated.

**Strengthened hook with holography:**

> *"Your three most celebrated inventions — CASSI, compressive holography, and gigapixel cameras — are all instances of the same 11 primitives. The same Gate 3 mismatch that limits CASSI reconstruction also limits holographic 3D recovery. We'd like your hardware expertise to validate this on physical instruments."*

**Updated contribution ask:**
1. Hardware mask displacement on CASSI/CACTI (existing package)
2. **NEW:** Validate compressive holography forward model specs
3. **NEW:** Provide holographic test data or review holography results
4. Manuscript review of optical imaging sections

**Effort for Brady:** 2–3 days lab time + 2–3 hours manuscript review

**Contribution package:** `contribution_packages/contribution_brady.md` (update with holography task)

---

### Prof. Steve Jiang (UT Southwestern) — INVITE WITH CBCT

**Profile:** Vice Chair of Digital Health & AI, Division Chief of Medical Physics & Engineering, UTSW Radiation Oncology. ~20,000 citations, h-index 76. Expert in CBCT, 4D-MRI, GPU Monte Carlo, PET-Linac.

**Current hook:** CT (parallel-beam) is validated.

**Strengthened hook with CBCT:**

> *"PWM diagnoses and corrects the calibration drift that dominates clinical QA failures across CT and CBCT — the two pillars of radiation oncology imaging. The Triad maps directly to clinical failure categories: Gate 1 = protocol inadequacy, Gate 2 = dose budget, Gate 3 = scanner calibration drift. We'd like your clinical imaging expertise to validate this on clinical CBCT."*

**Updated contribution ask:**
1. CT phantom CoR offset experiment (existing package)
2. **NEW:** CBCT scatter correction validation on clinical linac CBCT
3. **NEW:** Clinical interpretation of Triad-to-QA mapping
4. Manuscript review of medical imaging sections

**Effort for Jiang:** 1–2 days scanner time + 2–3 hours manuscript review

**Contribution package:** `contribution_packages/contribution_jiang.md` (update with CBCT task)

---

## 6. Before vs. After Comparison

| Dimension | Current Paper | After Adding 5 |
|-----------|--------------|----------------|
| Validated modalities | 7 | **12** |
| Carrier families validated | 4 of 5 (no acoustic) | **5 of 5** |
| Biology modalities | 0 | **2** (Cryo-EM, Fluorescence) |
| Clinical modalities | 2 (CT, MRI) | **4** (+CBCT, Ultrasound) |
| Coherent 3D optical | 0 | **1** (Holography) |
| Primitives tested | 11 (but P only via ptychography) | 11 (P tested in 3 modalities) |
| Nobel Prize modalities | 0 explicitly | **2** (Cryo-EM 2017, Super-res microscopy 2014) |
| Brady hook | CASSI only | CASSI + **Holography** (his Leith Medal work) |
| Jiang hook | CT only | CT + **CBCT** (his deepest research area) |
| Nature audience | Physics/engineering | **+Biology +Medicine** |
| Hardware validations | 5 | **5+** (extend with CBCT/ultrasound phantom) |
| Held-out closure test | 8 | 8+ (can expand) |
| Registered templates | 26 | **31+** |

### Carrier Balance After Adding 5

| Carrier Family | Current | After |
|---------------|---------|-------|
| Incoherent photons | 4 (CASSI, CACTI, SPC, Lensless) | 4 |
| Coherent photons | 0 optical | **1** (Holography) |
| Electrons | 1 (Ptychography) | **2** (+Cryo-EM) |
| X-ray photons | 1 (CT) | **2** (+CBCT) |
| Nuclear spins | 1 (MRI) | 1 |
| Acoustic waves | **0** | **1** (Ultrasound) |
| **Total** | **7** | **12** |

The portfolio is now balanced: no carrier has more than 4 modalities, every carrier is validated, and the two largest scientific audiences (biology, medicine) are covered.

---

## 7. Implementation Priority & Effort Estimates

### Priority Order

| Priority | Modality | Effort | Time | Blocks |
|----------|----------|--------|------|--------|
| **1** | **Ultrasound** | Low | 1–2 weeks | Fills acoustic gap (reviewer will flag) |
| **2** | **CBCT** | Low | 1–2 weeks | Extends CT, hooks Jiang |
| **3** | **Cryo-EM** | Medium | 2–3 weeks | Hooks Nature's biology audience |
| **4** | **Compressive Holography** | Medium | 2–3 weeks | Hooks Brady |
| **5** | **Fluorescence Microscopy** | Medium | 2–3 weeks | Deepens biology coverage |

### What Each Addition Requires

**For each new modality:**
1. OperatorGraph template (DAG definition + primitive parameters) — 1–2 days
2. Forward model implementation (forward + adjoint) — 2–3 days
3. Mismatch parameter database — 1 day
4. 4-Scenario validation (simulation) — 2–3 days
5. Results analysis + figures — 1–2 days
6. Paper text (methods, results, supplementary table rows) — 1 day

**Total per modality:** ~1–2 weeks for Low effort, ~2–3 weeks for Medium effort

### Minimum Viable Submission

If time is severely limited, add just **Ultrasound + CBCT** (both Low effort, ~2–3 weeks total). This:
- Fills the acoustic carrier gap (eliminates the most likely reviewer objection)
- Extends clinical coverage to 4 modalities
- Hooks Jiang
- Brings validated count to 9 across 5 carriers

### Full Recommended Submission

Add all 5 (~8–12 weeks total). This gives 12 validated modalities across 5 carriers with biology + medicine + physics coverage — definitively Nature-level breadth.

---

## 8. Paper Structure Improvements

### Abstract Changes
Add: "...spanning five carrier families — photons, electrons, X-rays, nuclear spins, **and acoustic waves** — including **cryo-electron microscopy** and **clinical cone-beam CT**..."

### Introduction Changes
Add one sentence: "The framework's scope extends from coded aperture cameras to clinical CBCT scanners to cryo-EM instruments used for drug discovery — demonstrating that the structural bottleneck in image recovery is universal across scientific imaging."

### Empirical Validation Section
Add subsections for each new modality (following the existing format):
- **Ultrasound:** Speed-of-sound mismatch, phase aberration correction, PICMUS benchmark
- **Cryo-EM:** CTF defocus mismatch, resolution vs. defocus error, EMPIAR benchmark
- **CBCT:** Scatter kernel mismatch + beam hardening, cupping artifact correction
- **Holography:** Propagation distance mismatch, twin-image suppression, 3D reconstruction
- **Fluorescence:** PSF mismatch, deconvolution improvement, ISBI benchmark

### Discussion Changes
Add smartphone paragraph: "The same framework applies to consumer devices: the 8 core features of smartphone computational photography (HDR, Night Mode, Portrait, Zoom, Stabilization, Multi-Camera Fusion, Color Science, Panorama) use 6 of the 11 primitives and exhibit Gate 3 dominance in 6 of 8 features under standard conditions (Supplementary Note XX)."

### Figures
- Update Fig. 9 (basis-growth saturation) to include the 5 new modalities at N=31–35
- Add new modalities to Extended Data Table 2 (modality registry)
- Add 1 panel to Fig. 4 (correction bar chart) showing new modalities

### Supplementary
- Add per-modality methods and results tables (following existing format)
- Add Supplementary Note: "Smartphone Computational Photography" with all 8 DAGs
- Add Supplementary Note: "Cryo-EM CTF Correction" with detailed derivation
