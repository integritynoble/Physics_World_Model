# PWM Nature Paper: Collaboration, Virality & Monetization Strategy

**Paper:** "Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging"
**Target:** Nature
**Date:** 2026-02-27

---

## Table of Contents

1. [Current Paper Status](#1-current-paper-status)
2. [Prof. Xin Yuan — Already Added](#2-prof-xin-yuan--already-added)
3. [Modalities to Attract Prof. David Brady](#3-modalities-to-attract-prof-david-brady)
4. [Modalities to Attract Prof. Steve Jiang](#4-modalities-to-attract-prof-steve-jiang)
5. [Modalities That Make PWM Go Viral](#5-modalities-that-make-pwm-go-viral)
6. [Modalities That Generate Revenue](#6-modalities-that-generate-revenue)
7. [Unified Recommendation for the Nature Paper](#7-unified-recommendation-for-the-nature-paper)
8. [Professor Profiles (Detailed)](#8-professor-profiles-detailed)
9. [Recommended Modalities to Add for Nature Submission](#9-recommended-modalities-to-add-for-nature-submission)

---

## 1. Current Paper Status

### Authors
1. **Chengshuai Yang** (corresponding) — NextGen PlatformAI C Corp, USA
2. **Xin Yuan** — School of Engineering, Westlake University, Hangzhou, China
3. ~~David J. Brady~~ (commented out, pending invitation)
4. ~~Steve B. Jiang~~ (commented out, pending invitation)

### Validated Modalities (7 fully validated, 4-Scenario Protocol)
| Modality | Carrier | Oracle Gain | Hardware Validated |
|---|---|---|---|
| CASSI | Photon | +0.76/+6.50 dB | Yes (TSA real data) |
| CACTI | Photon | +10.21 dB | Yes (EfficientSCI real data) |
| SPC | Photon | +7.71/+10.38 dB | No |
| Lensless | Photon | +3.55 dB | No |
| Ptychography | Electron | +7.09 dB | Yes (4D-STEM SrTiO₃) |
| CT | X-ray | +10.68 dB | Yes (FIPS walnut, HTC sinograms) |
| MRI | Spin | +1.75 to +7.14 dB | Yes (M4Raw multi-coil) |

### Held-out Closure Test (8 modalities, all pass)
OCT, Photoacoustic, SIM, Phase-contrast X-ray, Electron Ptychography, Ghost Imaging, THz-TDS, Compton Scatter

### Current Carrier Coverage
5 families: Photons, Electrons, Spins, Acoustic waves, Particles

---

## 2. Prof. Xin Yuan — Already Added

**Status:** ✅ Already listed as co-author #2 in `main.tex` (line 9)

**Profile:**
- Associate Professor, School of Engineering, Westlake University, Hangzhou, China
- Previously: Video Analysis and Coding Lead Researcher, Bell Labs, Murray Hill, NJ (2015–2021)
- Postdoc at Duke University (2012–2015) working with **David Brady** on compressive sensing
- PhD: The Hong Kong Polytechnic University (2012)
- Directs the **Sensing and Computational Imaging (SCI) Lab** at Westlake
- Awards: National Excellent Young Scholar (overseas, 2021); Distinguished Young Scholar of Zhejiang Province (2022)
- ~15,600+ citations, 70+ journal papers, 60+ conference papers

**Why He's the Right Co-author:**
- He is arguably **the central figure** in snapshot compressive imaging (SCI) today
- Authored the definitive survey: "Snapshot Compressive Imaging: Theory, Algorithms and Applications" (IEEE Signal Processing Magazine, 2021)
- Unified CASSI, CACTI, and SPC under a single theoretical framework (the SCI framework: `y = H * x + noise`)
- Developed **GAP-TV** (the primary solver used across CASSI, CACTI, and SPC experiments in the paper)
- Contributed **EfficientSCI** architecture used for CACTI validation (CVPR 2023)
- Worked directly on CASSI at Duke University with David Brady
- Developed block-wise lensless compressive camera at Bell Labs (directly relates to `spc_block` benchmark)
- The PWM codebase has deep ties to his work: `cassi_operator.py`, `cacti_operator.py`, `spc_operator.py`, `gap_tv.py`, `efficientsci.py`, `mst.py`

**Key Publications:**
1. "Snapshot Compressive Imaging: Theory, Algorithms and Applications" — IEEE SPM 2021 (with Brady & Katsaggelos)
2. "Rank Minimization for Snapshot Compressive Imaging" (DeSCI) — IEEE TPAMI 2019
3. "Plug-and-Play Algorithms for Large-scale Snapshot Compressive Imaging" — CVPR 2020
4. "EfficientSCI: Densely Connected Network..." — CVPR 2023
5. "Block-wise Lensless Compressive Camera" — Bell Labs, 2017
6. "Single-Pixel Neutron Imaging with AI" — The Innovation, 2021

**Contribution Package:** `contribution_packages/contribution_yuan.md` (ready)
- Task 1: Validate reconstruction algorithm parameters (GAP-TV, EfficientSCI, PnP-FFDNet, ELP-Unfolding)
- Task 2: Validate CASSI/CACTI forward model specifications (5-parameter mismatch model)
- Task 3: Review real-data experimental protocol (TSA + EfficientSCI datasets)
- Task 4: Manuscript review (empirical validation sections, methods, supplementary tables)
- Estimated effort: 2–3 days

**No changes needed.** The author listing and contribution package are complete.

---

## 3. Modalities to Attract Prof. David Brady

### Brady's Profile Summary

**Position:** J.W. and H.M. Goodman Endowed Chair in Optical Sciences, Wyant College of Optical Sciences, University of Arizona (since 2021). Previously Michael J. Fitzpatrick Professor of ECE at Duke University.

**Stature:** Fellow of Optica, IEEE, and SPIE. ~594 publications, ~17,600+ citations, h-index ~60–70.

**Key Honors:**
- Optica Emmett N. Leith Medal (2023) — for the invention of sparse holography
- SPIE Dennis Gabor Award (2013)
- Author of textbook *Optical Imaging and Spectroscopy* (Wiley, 2009)

### What Brady Has Invented/Pioneered

| Modality | Brady's Role | Notable Paper | Citations |
|---|---|---|---|
| **CASSI** (Coded Aperture Snapshot Spectral Imaging) | Inventor/pioneer | "Single disperser design for CASSI" | ~529 |
| **Gigapixel / Multi-aperture Array Camera** (AWARE) | Pioneer | "Multiscale gigapixel photography" — **Nature 2012** | Very high |
| **Compressive Holography** | Inventor | "Compressive Holography" — Opt. Express 2009 | High |
| **Coded Aperture X-ray Scatter Tomography** | Lead developer | "Coded apertures for x-ray scatter imaging" | Moderate |
| **Millimeter-wave / THz Computational Imaging** | Lead developer | "Large Metasurface Aperture for MMW CI" — Sci. Rep. 2017 | Moderate |
| **Aperture Synthesis / Interferometric Imaging** | Current primary focus | "Multiscale aperture synthesis imager" — **Nature Comms 2025** | Recent |

### Current Attraction Points in the Paper
- CASSI is already validated (his most famous invention) ✅
- CACTI is already validated ✅
- Contribution package asks for hardware mask displacement experiments ✅

### Modalities to Add — Must-Add (High Priority)

#### A. Compressive Holography ⭐⭐⭐

**Why this is the killer feature for Brady:**
- He won the **2023 Optica Emmett N. Leith Medal** specifically for inventing sparse holography
- Reconstructing 3D volumes from single 2D holograms using sparsity priors
- This is his most personally prestigious work (a named medal from the top optics society)

**OperatorGraph Primitive Chain:**
```
Source → P(d,λ) → M(object) → P(d',λ) → D(|·|²)
```
- `P` (Propagate): Free-space wave propagation (reference and object beams)
- `M` (Modulate): Object interaction (amplitude + phase modulation)
- `D` (Detect): Intensity square-law detection (hologram recording)

**Gate 3 Mismatch Parameters:**
- Propagation distance error (defocus)
- Reference beam angle error
- Wavelength calibration error
- Twin-image contamination (incomplete twin-image suppression)

**Effort to Add:** Medium — coherent wave propagation already exists in the Ptychography template. The `P` primitive handles Fresnel/Fraunhofer propagation.

**Testable Prediction:** Propagation distance mismatch of 1% should produce +3–8 dB correction gain. Twin-image suppression via correct forward model should outperform algorithmic twin-image removal.

**Impact on Brady:** Shows that his Leith Medal invention is an instance of the same 11 primitives. He becomes personally invested.

#### B. Gigapixel / Multi-aperture Array Camera ⭐⭐⭐

**Why this matters for Brady:**
- His **Nature 2012 paper** — the AWARE program that built the world's first gigapixel camera
- This is his highest-profile publication and his identity-defining project

**OperatorGraph Primitive Chain:**
```
For each sub-aperture i:
  Source → C(h_i) → S(Ω_i) → D(g_i)
Then: Σ (computational fusion across sub-apertures)
```
- `C` (Convolve): Per-aperture PSF (shift-variant across the array)
- `S` (Sample): Field-of-view selection per micro-camera
- `Σ` (Accumulate): Computational image fusion
- `D` (Detect): Per-camera detector response

**Gate 3 Mismatch Parameters:**
- Per-camera focal length variation
- Inter-camera registration error (sub-pixel alignment between overlapping FOVs)
- Per-camera PSF variation (manufacturing tolerance of micro-optics)
- Geometric distortion mismatch across the array

**Effort to Add:** Medium — all primitives exist. New template needed for the array topology.

**Testable Prediction:** Inter-camera registration mismatch of 0.5 px should cause visible seam artifacts. PWM correction should achieve seamless fusion by correcting per-camera geometry parameters.

**Impact on Brady:** His Nature paper is validated by your framework. This is the single most compelling pitch you can make.

### Modalities to Add — Nice-to-Add (Lower Priority)

#### C. Coded Aperture X-ray Scatter Tomography

**Why:** His CAXI program at Duke. Extends coded apertures from optical to X-ray domain. Uses the `R` (Scatter) primitive added via the extension protocol.

**Primitive Chain:** `Π → R → M → D` (projection + scatter + coded aperture + detection)

**Effort:** Low — variant of CT + coded aperture. Scatter primitive already in library.

#### D. Millimeter-wave / THz Computational Imaging

**Why:** His metasurface aperture work at Duke. Different carrier family (RF/THz).

**Effort:** Medium — new carrier but existing primitives should suffice.

#### E. Aperture Synthesis / Interferometric Imaging

**Why:** His *current* active focus at Arizona (Nature Communications 2025). Would make him invested in PWM's future.

**Primitive Chain:** `P → S → F → D` (propagation + baseline sampling + Fourier encoding + detection)

**Effort:** Higher — interferometric baseline processing needs careful template design.

### The Brady Pitch (Recommended Framing)

> *"Your three most celebrated inventions — CASSI, compressive holography, and gigapixel cameras — are all instances of the same 11 primitives. The same Gate 3 mismatch that limits CASSI reconstruction also limits holographic 3D recovery and gigapixel image fusion. We'd like your hardware expertise to validate this on physical instruments."*

**Updated contribution ask:**
1. Hardware mask displacement on CASSI/CACTI (existing ask) ✅
2. **NEW:** Validate compressive holography forward model specs and provide hologram test data
3. **NEW:** Validate gigapixel camera forward model specs (can use existing AWARE data)
4. Manuscript review of optical imaging sections

---

## 4. Modalities to Attract Prof. Steve Jiang

### Jiang's Profile Summary

**Position:** Vice Chair of Digital Health & AI, Division Chief of Medical Physics & Engineering, Professor, Department of Radiation Oncology, UT Southwestern Medical Center. Holds the David A. Pistenmaa, M.D., Ph.D. Distinguished Chair. Directs the MAIA Lab (Medical AI and Automation Laboratory).

**Stature:** ~20,000 citations, h-index 76, 516+ publications. Fellow of AAPM, AIMBE, IoP.

**Key Research Areas:**
- AI/Deep Learning in medical imaging
- GPU-accelerated Monte Carlo dose calculation (pioneered gDPM, gPMC, goMC)
- Adaptive radiation therapy (ART) and image-guided radiotherapy
- 4D imaging and respiratory motion management (co-authored AAPM TG-76)
- Biology-guided radiation therapy (BgRT) with PET-Linac

### What Jiang Has Pioneered

| Modality | Jiang's Role | Notable Paper | Citations |
|---|---|---|---|
| **CBCT** (Cone-Beam CT) | Deep expertise: low-dose, 4D, scatter correction, CycleGAN synthesis | CycleGAN CBCT-to-CT (Roberts' Prize nominee) | Very high |
| **CT** (standard, spectral, sparse-view) | Compressed-sensing reconstruction, spectral CT | Multiple papers | High |
| **4D-MRI** | SK-MEC method (motion-compensated volumetric MRI from undersampled k-space) | SK-MEC paper | Moderate |
| **PET/CT** (PET-Linac) | Biology-guided radiotherapy with RefleXion X1 | BgRT clinical studies | Growing |
| **4D-CT** | Respiratory motion management | AAPM TG-76 report | ~1,562 |
| **Monte Carlo simulation** | GPU-accelerated radiation transport (gDPM, gPMC, goMC) | gDPM paper | High |

### Current Attraction Points in the Paper
- CT (parallel-beam) is validated ✅
- Contribution package asks for ACR phantom CoR offset experiment ✅

### Modalities to Add — Must-Add (High Priority)

#### A. Cone-Beam CT (CBCT) ⭐⭐⭐

**Why this is the killer feature for Jiang:**
- CBCT is his **most prolific and deeply-invested** research area
- He has published on: low-dose CBCT reconstruction, 4D-CBCT, scatter correction, Bio-4DCBCT, CBCT-to-CT synthesis via CycleGAN, GPU-based iterative reconstruction
- CBCT is the **workhorse of image-guided radiation therapy** (IGRT) — used before every radiation treatment fraction
- There are ~10,000+ CBCT-equipped linear accelerators in the US alone

**OperatorGraph Primitive Chain:**
```
Source → Π(θ, cone geometry) → R(scatter kernel) → D(flat-panel)
```
or equivalently:
```
Source → Π(θ) → C(scatter PSF) → D(g, η_linear)
```
- `Π` (Project): Cone-beam geometry projection (extension of parallel-beam)
- `R` or `C`: Scatter contamination (convolution with scatter kernel or scatter interaction)
- `D` (Detect): Flat-panel detector response (gain nonuniformity)

**Gate 3 Mismatch Parameters (clinically relevant):**
- **Scatter kernel mismatch** — the scatter distribution is patient-dependent and never perfectly known; this is the #1 image quality limitation in CBCT
- **Geometric calibration drift** — source-detector alignment drifts over weeks/months
- **Detector gain variation** — flat-panel detector elements degrade non-uniformly
- **Beam hardening** — polyenergetic X-ray spectrum vs. monoenergetic model assumption (uses `Λ` Transform primitive)

**Effort to Add:** Low — extends the existing CT template from parallel-beam to cone-beam geometry. The `Π` projection primitive already supports angular parameterization.

**Testable Prediction:** Scatter kernel mismatch should produce cupping artifacts and HU inaccuracy. PWM scatter correction should recover +3–6 dB over uncorrected CBCT. Geometric calibration drift of 1–2 mm should cause +2–4 dB degradation with 100% oracle recovery.

**Clinical Impact:** Every radiation oncology department fights CBCT image quality issues daily. Automated mismatch detection saves medical physicist time and catches calibration drift before it affects treatment.

#### B. 4D-MRI ⭐⭐

**Why this matters for Jiang:**
- He developed the **SK-MEC method** (k-space-driven Motion Estimation and Compensation) for volumetric MRI from undersampled data
- 4D-MRI is critical for radiation therapy planning — it tracks tumor motion during breathing
- Directly connects to the existing MRI template with a temporal dimension

**OperatorGraph Primitive Chain:**
```
Source → M(coil sensitivities) → F(k-space trajectory) → S(undersampling) → Σ(temporal) → D
```
- Extension of existing MRI chain with `Σ` (Accumulate) over the temporal/respiratory dimension
- Motion states are parameterized as additional mismatch dimensions

**Gate 3 Mismatch Parameters:**
- Coil sensitivity mismatch (same as standard MRI)
- Motion model error (assumed vs. actual respiratory trajectory)
- k-space trajectory deviation (gradient imperfections during rapid acquisition)

**Effort to Add:** Low — variant of existing MRI template with temporal accumulation.

**Impact on Jiang:** Shows PWM handles the temporal dimension that is critical for his radiation therapy imaging work.

#### C. PET (Positron Emission Tomography) ⭐⭐

**Why this matters for Jiang:**
- He is actively working on **biology-guided radiation therapy (BgRT)** with the RefleXion X1 PET-Linac
- PET reconstruction from limited time-of-flight data is a classic inverse problem
- PET is a fundamentally different detection modality (coincidence detection of annihilation photons)

**OperatorGraph Primitive Chain:**
```
Source → Π(line-of-response geometry) → D(Poisson counting, coincidence)
```
- `Π` (Project): Line-of-response (LOR) projection geometry
- `D` (Detect): Poisson photon-counting statistics with coincidence timing

**Gate 3 Mismatch Parameters:**
- **Attenuation map mismatch** — PET requires CT-derived attenuation correction; errors in the attenuation map directly cause quantitative errors
- **Detector timing resolution** — time-of-flight (TOF) bin width affects spatial resolution
- **Scatter fraction** — scatter correction models are approximate
- **Random coincidence rate** — estimated but imperfect subtraction

**Effort to Add:** Medium — new template. The `Π` projection primitive already exists; main addition is the Poisson detection model and coincidence geometry.

**Testable Prediction:** Attenuation map mismatch of 5% should cause quantitative PET error of 8–15%. This directly impacts SUV accuracy in radiation therapy planning.

**Clinical Impact:** PET-guided radiation therapy is the frontier of precision oncology. PWM calibration of the PET forward model has direct clinical translation.

### Modalities to Add — Nice-to-Add (Lower Priority)

#### D. Cross-Modality Synthesis (CBCT → CT)

**Why:** Jiang's CycleGAN CBCT-to-CT work. PWM could diagnose *why* domain gaps exist between modalities (Gate 3 mismatch between the CBCT forward model and CT forward model).

**Conceptual contribution** — doesn't require new primitives but shows the Triad applies to cross-domain translation problems.

#### E. GPU Monte Carlo Dose Simulation

**Why:** His gDPM/gPMC/goMC packages simulate radiation transport physics. PWM's operator graph could model particle transport chains.

**Effort:** Higher — stretch goal for a future extension.

### The Jiang Pitch (Recommended Framing)

> *"The Triad Decomposition diagnoses and corrects the calibration drift that dominates clinical QA failures across CT, CBCT, MRI, and PET — the four pillars of radiation oncology imaging. We'd like your clinical imaging expertise to validate this on clinical scanners. The CT QC Copilot concept maps directly to the ACR accreditation workflow your department performs annually."*

**Updated contribution ask:**
1. CT phantom CoR offset experiment (existing ask) ✅
2. **NEW:** CBCT scatter correction validation on a clinical linac CBCT
3. **NEW:** Provide 4D-MRI test case (SK-MEC undersampled k-space + ground truth)
4. **NEW (optional):** PET attenuation correction validation on PET-Linac
5. Clinical interpretation of Triad-to-QA mapping
6. Manuscript review of medical imaging sections

---

## 5. Modalities That Make PWM Go Viral

### Tier 1 — Massive Audience, Immediate Viral Potential

#### A. Smartphone Computational Photography ⭐⭐⭐ (MOST VIRAL)

**Why viral:** Every person with a phone. Show that iPhone/Android camera pipelines (HDR, night mode, portrait depth, computational zoom) are instances of the same 11 primitives.

**Audience:** ~4 billion smartphone users. Tech press covers this obsessively.

**Primitive Chain (Night Mode example):**
```
Source → C(PSF_motion) → Σ(multi-frame burst) → M(HDR tone map) → D
```

**Primitive Chain (Portrait Mode / computational bokeh):**
```
Source → P(depth estimation) → C(PSF_depth) → M(segmentation mask) → D
```

**Viral angle:** "Your iPhone's portrait mode and your hospital's MRI scanner use the same 11 physics primitives." This is a Nature-worthy headline.

**Effort:** Medium — familiar hardware, but need to formalize the computational photography pipeline as OperatorGraph templates.

#### B. Medical CT/MRI — "Your Scanner Is Solving the Wrong Problem" ⭐⭐⭐

**Why viral:** Already partially in the paper. The headline "operator mismatch, not algorithm weakness, is the bottleneck" is provocative and counterintuitive to the radiology and AI communities.

**Audience:** ~100,000 radiologists + millions of patients + the entire medical AI research community.

**Viral angle:** "The billions spent on AI reconstruction algorithms are targeting the wrong bottleneck. Calibration — not deep learning — is the underinvested lever."

**Effort:** Already done — just need to amplify the clinical narrative.

#### C. Autonomous Driving Sensors (LiDAR + Radar + Camera Fusion) ⭐⭐

**Why viral:** Massive VC/industry attention. Sensor fusion = multi-modal operator graph. Every autonomous vehicle processes multiple imaging modalities simultaneously.

**Audience:** Tesla, Waymo, Cruise, Mobileye, the entire AV industry, tech press, investors.

**Primitive Chain (LiDAR):**
```
Source → P(time-of-flight) → S(scanning pattern) → D(SPAD array)
```

**Primitive Chain (Radar):**
```
Source → F(chirp encoding) → Σ(Doppler integration) → D
```

**Viral angle:** "The same mismatch that ruins your MRI also ruins your self-driving car's LiDAR." Cross-domain unification is inherently newsworthy.

**Effort:** Medium-high — need LiDAR and radar templates. New carrier (RF for radar) but primitives should suffice.

#### D. Cryo-EM (Structural Biology) ⭐⭐⭐

**Why viral:** The tool behind AlphaFold-era drug discovery. Nobel Prize modality (2017 Chemistry: Jacques Dubochet, Joachim Frank, Richard Henderson). Massive structural biology community.

**Audience:** ~50,000 structural biologists + pharma industry + anyone who follows Nobel-adjacent science.

**Primitive Chain:**
```
Source → P(electron wave) → M(specimen) → P(objective lens) → C(CTF) → D(direct electron detector)
```

**Gate 3 Mismatch:** CTF (Contrast Transfer Function) estimation is the critical calibration step. Defocus mismatch directly limits resolution. This is *exactly* the Gate 3 problem.

**Viral angle:** "The same framework that calibrates a $500 camera also calibrates a $5M cryo-EM — and the bottleneck is the same: operator mismatch."

**Effort:** Medium — electron optics template shares primitives with optical ptychography.

### Tier 2 — Strong Scientific Virality

#### E. James Webb Space Telescope (JWST) Imaging ⭐⭐

**Why viral:** Most famous telescope in the world. Its wavefront sensing and mirror alignment pipeline is a Gate 3 correction problem.

**Audience:** Astronomy community + general public fascinated by JWST images.

**Primitive Chain:**
```
Source → P(d, λ) → M(segmented mirror) → C(PSF) → D(NIRCam/MIRI)
```

**Gate 3 Mismatch:** Mirror segment alignment error, PSF variation across field.

**Viral angle:** "The $10B telescope's image quality depends on the same forward-model calibration that governs a coded aperture camera."

**Effort:** Medium — wavefront sensing template uses existing primitives.

#### F. Gravitational Wave Detection (LIGO) ⭐

**Why viral:** Enormous physics audience. LIGO's matched filtering is fundamentally template matching against a forward model — Gate 3 applies when the waveform template mismatches the true signal.

**Audience:** Physics community (hundreds of thousands), general public who follow LIGO discoveries.

**Viral angle:** Provocative but requires careful framing. Best as a discussion-section mention rather than full validation.

**Effort:** High — very different from imaging. Best as a conceptual extension.

#### G. Brain MRI (fMRI, Diffusion MRI) ⭐⭐

**Why viral:** Neuroscience is huge. The Human Connectome Project, Brain Initiative, and all fMRI-based research depend on MRI reconstruction quality.

**Audience:** ~100,000+ neuroscience researchers.

**Primitive Chain:** Extension of existing MRI template with BOLD contrast model (fMRI) or diffusion encoding (dMRI).

**Effort:** Low — variant of existing MRI template.

### Virality Summary Table

| Modality | Audience Size | Headline Appeal | Effort | Viral Score |
|---|---|---|---|---|
| Smartphone photography | 4B users | ⭐⭐⭐⭐⭐ | Medium | **10/10** |
| Medical CT/MRI narrative | 100k+ radiologists | ⭐⭐⭐⭐ | Already done | **9/10** |
| Cryo-EM | 50k structural biologists | ⭐⭐⭐⭐ | Medium | **9/10** |
| Autonomous driving sensors | Entire AV industry | ⭐⭐⭐⭐ | Medium-high | **8/10** |
| JWST | Astronomy + public | ⭐⭐⭐⭐⭐ | Medium | **8/10** |
| Brain MRI (fMRI/dMRI) | 100k neuroscientists | ⭐⭐⭐ | Low | **7/10** |
| LIGO | Physics community | ⭐⭐⭐⭐ | High | **6/10** |

---

## 6. Modalities That Generate Revenue

### Tier 1 — Highest Revenue Potential

#### A. Clinical CT/CBCT QA SaaS ⭐⭐⭐ ($$$)

**Revenue Model:** SaaS subscription per scanner per year.

**Market Size:** ~$2B CT quality assurance market. ~50,000 CT scanners + ~10,000 CBCT-equipped linacs in the US alone.

**Pricing:** $5,000–$15,000/scanner/year (comparable to existing QA software like Sun Nuclear, Standard Imaging).

**Why They Pay:**
- Hospitals are *legally required* to perform annual ACR accreditation and routine QA
- Medical physicists spend 2–4 hours per scanner per month on QA — this is expensive labor
- Automated mismatch detection catches calibration drift *before* accreditation failure
- The CT QC Copilot concept is already half-built in the paper
- CoR drift, HU drift, gain variation — all Gate 3 problems with automated solutions

**Competitive Advantage:** No existing QA software diagnoses *why* a metric fails. PWM doesn't just detect that uniformity is bad — it identifies CoR offset as the root cause and quantifies the correction.

**Path to Revenue:**
1. Ship CT QC Copilot as standalone SaaS product
2. Partner with Jiang's department for clinical validation
3. Sell to medical physics departments through AAPM channels
4. Expand to CBCT QA for radiation therapy departments

**Revenue Estimate:** 1,000 scanners × $10k/yr = **$10M ARR** (conservative early adoption)

#### B. Semiconductor Inspection ⭐⭐⭐ ($$$$$)

**Revenue Model:** Per-tool license or per-wafer SaaS.

**Market Size:** ~$15B semiconductor inspection and metrology market (KLA, ASML, Applied Materials dominate).

**Pricing:** $100,000–$500,000 per tool license. Semiconductor companies have enormous budgets for yield improvement.

**Why They Pay:**
- Sub-nanometer calibration mismatch = dead chips = millions in lost yield
- Electron ptychography is already validated in the paper (4D-STEM SrTiO₃)
- EUV lithography metrology needs better forward-model calibration
- Every wafer fab has 100+ inspection tools, each needing continuous calibration
- A 0.1% yield improvement on a $20B fab = $20M/year savings

**Key Modalities:**
- **Electron ptychography** (already validated) — used for atomic-resolution semiconductor metrology
- **SEM** (Scanning Electron Microscopy) — the workhorse of semiconductor inspection
- **EUV metrology** — ASML's lithography systems need forward-model calibration

**Path to Revenue:**
1. Partner with one semiconductor equipment maker (KLA, ASML, Hitachi High-Tech)
2. Demonstrate yield improvement on a production line
3. License per-tool

**Revenue Estimate:** 100 tool licenses × $200k/yr = **$20M ARR**

#### C. MRI QA / Calibration SaaS ⭐⭐ ($$$)

**Revenue Model:** SaaS subscription per scanner.

**Market Size:** ~$7B MRI market. ~40,000 MRI scanners in the US.

**Pricing:** $8,000–$20,000/scanner/year.

**Why They Pay:**
- MRI calibration is complex (coil sensitivities, B0/B1 mapping, gradient calibration)
- ESPIRiT (the standard coil sensitivity method) degrades under data-limited conditions (the paper shows -9.29 dB at 24 ACS lines)
- PWM's cross-modality approach could outperform for multi-vendor hospital fleets
- Quantitative MRI (qMRI) demands accurate forward models

**Revenue Estimate:** 500 scanners × $12k/yr = **$6M ARR**

### Tier 2 — Strong Revenue with Lower Barrier to Entry

#### D. Defense/Intelligence (SAR, Hyperspectral) ⭐⭐ ($$$$)

**Revenue Model:** Government contracts (SBIR, STTR, direct procurement).

**Why They Pay:**
- NGA, NRO, DARPA pay premium for automated sensor calibration
- SAR autofocus = Gate 3 correction (platform motion error → phase error in SAR image)
- Hyperspectral target detection degrades with uncalibrated spectral response
- Export-controlled, high-margin, long contract cycles

**Key Modalities:** SAR, hyperspectral (CASSI variant), infrared imaging

**Revenue Estimate:** 2–3 government contracts × $2–5M each = **$5–15M**

#### E. Microscopy Deconvolution Plugin ⭐⭐ ($$)

**Revenue Model:** Per-seat license or institutional subscription.

**Market Size:** ~$8B microscopy market. Thousands of core imaging facilities worldwide.

**Pricing:** $3,000–$10,000 per seat (comparable to Huygens deconvolution at ~$5k/seat).

**Why They Pay:**
- Every confocal, widefield, and super-resolution microscope needs PSF correction
- Current deconvolution software (Huygens, AutoQuant) uses fixed PSF models — Gate 3 mismatch!
- PWM-based adaptive PSF estimation corrects for sample-induced aberrations
- Core facilities serve 50–200 researchers each

**Key Modalities:** Confocal, widefield fluorescence, light-sheet, super-resolution (STED, PALM, STORM)

**Path to Revenue:** Ship as ImageJ/Fiji plugin or standalone software. Sell through Nikon, Zeiss, Leica partnerships or directly to core facilities.

**Revenue Estimate:** 500 seats × $5k/yr = **$2.5M ARR** (grows with microscopy market)

#### F. Industrial NDT (Non-Destructive Testing) X-ray CT ⭐⭐ ($$$)

**Revenue Model:** Per-system license.

**Why They Pay:**
- Aerospace (Boeing, Airbus) and automotive (Tesla, BMW) use X-ray CT for defect detection
- Miscalibrated CT = missed defects = liability (aircraft engine failure, battery pack failure)
- CT is already validated in the paper — direct transfer

**Revenue Estimate:** 200 systems × $20k/yr = **$4M ARR**

#### G. Cryo-EM CTF Estimation ⭐⭐ ($$$)

**Revenue Model:** SaaS for structural biology labs.

**Why They Pay:**
- CTF estimation and correction is the rate-limiting calibration step in cryo-EM workflows
- Errors in CTF estimation directly limit resolution (the difference between 3Å and 2Å can determine publishability)
- Labs would pay to automate and improve CTF fitting
- Pharma companies running cryo-EM for drug discovery have large budgets

**Key Modalities:** Single-particle cryo-EM, cryo-electron tomography

**Revenue Estimate:** 100 labs × $15k/yr = **$1.5M ARR** (high-value customers)

### Revenue Summary Table

| Modality / Product | Revenue Model | Year 1–2 ARR Estimate | Barrier to Entry | Time to Revenue |
|---|---|---|---|---|
| **Clinical CT/CBCT QA SaaS** | Subscription/scanner | $10M | Medium (FDA pathway) | 12–18 months |
| **Semiconductor Inspection** | Per-tool license | $20M | High (partnership needed) | 18–24 months |
| **MRI QA SaaS** | Subscription/scanner | $6M | Medium | 12–18 months |
| **Defense (SAR, Hyperspectral)** | Government contracts | $5–15M | Medium (clearance) | 6–12 months |
| **Microscopy Plugin** | Per-seat license | $2.5M | Low (fastest to ship) | 3–6 months |
| **Industrial NDT CT** | Per-system license | $4M | Low–Medium | 6–12 months |
| **Cryo-EM CTF** | Lab SaaS | $1.5M | Medium | 12 months |

### Fastest Path to Revenue (Recommended Priority)

1. **Microscopy deconvolution plugin** — Ship in 3–6 months. Low regulatory barrier. Immediate market.
2. **Clinical CT/CBCT QA SaaS** — Leverage Jiang connection. CT QC Copilot is half-built. Clear regulatory pathway (QA software, not diagnostic).
3. **Defense contracts (SAR)** — SBIR/STTR grants provide non-dilutive funding while building the product.
4. **Semiconductor inspection** — Highest per-unit value but requires partnership. Start conversations with KLA/ASML now.

---

## 7. Unified Recommendation for the Nature Paper

### What to Add for Maximum Impact

Considering all four goals (attract Brady, attract Jiang, virality, revenue potential), here is the prioritized list of modalities to add:

| Priority | Modality | Attracts | Viral | Revenue | Effort |
|---|---|---|---|---|---|
| **1** | **Compressive Holography** | Brady ⭐⭐⭐ | Medium | Medium | Medium |
| **2** | **CBCT (Cone-Beam CT)** | Jiang ⭐⭐⭐ | Medium | $$$$ | Low |
| **3** | **Cryo-EM** | Nature reviewers | ⭐⭐⭐⭐ | $$$ | Medium |
| **4** | **Gigapixel Camera** | Brady ⭐⭐⭐ | High | Low | Medium |
| **5** | **PET** | Jiang ⭐⭐ | Medium | $$$ | Medium |
| **6** | **Smartphone Photography** | Nobody specific | ⭐⭐⭐⭐⭐ | Low direct | Medium |
| **7** | **4D-MRI** | Jiang ⭐⭐ | Medium | $$$ | Low |

### Recommended Action Plan

**Phase 1 (Immediate — for Nature submission):**
- Add **Compressive Holography** and **Gigapixel Camera** templates → invite Brady
- Add **CBCT** and **PET** templates → invite Jiang
- Add **Cryo-EM** template → strengthens Nature appeal (biology audience)
- Update held-out closure test: add all 5 new modalities to show basis completeness at N=35+

**Phase 2 (Post-acceptance, for press/outreach):**
- Add **Smartphone Photography** example to discussion section or supplementary
- Add **JWST** conceptual analysis to "beyond imaging" paragraph
- These drive media coverage and public interest

**Phase 3 (Post-publication, for revenue):**
- Ship **Microscopy plugin** (fastest revenue)
- Launch **CT/CBCT QA SaaS** (highest medical revenue, leverages Jiang)
- Pursue **Semiconductor partnerships** (highest total revenue)

### Updated Paper Modality Count

After Phase 1:
- **12 fully validated modalities** (up from 7): +Compressive Holography, +Gigapixel, +CBCT, +PET, +Cryo-EM
- **13 held-out closure test** (up from 8): +4D-MRI, +SAR, +Smartphone, +JWST wavefront sensing, +Diffusion MRI
- **35+ registered templates** (up from 26)
- **5 carrier families** (unchanged, but deeper coverage)
- **6+ hardware validations** (up from 5)

This scope — 12 validated modalities across 5 carriers with 35+ templates — is definitively Nature-level breadth.

---

## 8. Professor Profiles (Detailed)

### Prof. David J. Brady

**Affiliation:** Wyant College of Optical Sciences, University of Arizona
**Chair:** J.W. and H.M. Goodman Endowed Chair in Optical Sciences
**Website:** https://www.optics.arizona.edu/person/david-brady
**Lab:** The Camera Lab (https://wp.optics.arizona.edu/cameralab/)

**Education & Career:**
- Previously: Michael J. Fitzpatrick Professor of ECE, Duke University
- Director, Duke Imaging and Spectroscopy Program (DISP)
- Author: *Optical Imaging and Spectroscopy* (Wiley, 2009) — foundational textbook

**Metrics:** ~594 publications, ~17,600+ citations, h-index ~60–70

**Fellowships:** Optica Fellow (2003), SPIE Fellow (2007), IEEE Fellow (2009)

**Awards:**
- Optica Emmett N. Leith Medal (2023) — invention of sparse holography
- SPIE Dennis Gabor Award (2013)

**Research Philosophy:** The estimation of high-dimensional objects from low-dimensional measurements. Build a physically accurate forward model, then use computational inference (compressive sensing, neural networks, Bayesian methods) to reconstruct. This is *exactly* the PWM philosophy.

**Most Cited Works:**
1. "Multiscale gigapixel photography" — **Nature** 486, 386 (2012)
2. "Single disperser design for coded aperture snapshot spectral imaging" (~529 citations)
3. "Single-shot compressive spectral imaging with a dual-disperser architecture" (~412 citations)
4. "Compressive coded aperture spectral imaging: An introduction" — IEEE SPM 2014 (~304 citations)
5. "Compressive Holography" — Optics Express 2009
6. "Snapshot Compressive Imaging: Theory, Algorithms and Applications" — IEEE SPM 2021 (with Yuan & Katsaggelos)

**Current Research Focus (2024–2026):**
- Aperture synthesis / interferometric imaging (Nature Communications 2025)
- Snapshot ptychographic wavefront camera arrays (Optics Express 2025)
- Interferometric focal planes (Optics Express 2025)

**Connection to PWM:**
- Invented CASSI (already validated)
- Co-authored the SCI survey with Xin Yuan (already a co-author)
- Forward-model-based inverse problems are his career theme
- His hardware expertise provides the controlled validation the paper needs

---

### Prof. Steve B. Jiang

**Affiliation:** Department of Radiation Oncology, UT Southwestern Medical Center
**Chair:** David A. Pistenmaa, M.D., Ph.D. Distinguished Chair in Radiation Oncology
**Roles:** Vice Chair of Digital Health & AI; Division Chief of Medical Physics & Engineering
**Website:** https://profiles.utsouthwestern.edu/profile/150563/steve-jiang.html
**Lab:** MAIA Lab (Medical Artificial Intelligence and Automation Laboratory)

**Metrics:** ~20,000 citations, h-index 76, 516+ publications

**Fellowships:** AAPM Fellow, AIMBE Fellow, IoP Fellow

**Most Cited Works:**
1. "The management of respiratory motion in radiation oncology: Report of AAPM Task Group 76" (~1,562 citations)
2. "Effects of intra-fraction motion on IMRT dose delivery" (~456 citations)
3. "The management of imaging dose during image-guided radiotherapy: Report of AAPM TG 75" (~401 citations)
4. "Generating synthesized CT from CBCT using CycleGAN" (Roberts' Prize nominee)
5. GPU-based fast Monte Carlo simulation packages (gDPM, gPMC, goMC)

**Research Areas:**
- AI/Deep Learning in medical imaging (synthetic image generation, reconstruction, auto-segmentation)
- GPU-accelerated Monte Carlo dose calculation (pioneered 40+ GPU toolkits)
- Adaptive radiation therapy (ART) with online re-planning
- 4D imaging and respiratory motion management
- Biology-guided radiation therapy (BgRT) with PET-Linac (RefleXion X1)

**Connection to PWM:**
- CT is already validated (his department does CT QA daily)
- CBCT is his deepest research area — perfect for expansion
- 4D-MRI (SK-MEC method) connects to existing MRI template
- PET-Linac (BgRT) is his cutting-edge work
- Clinical QA narrative maps perfectly to the Triad (Gate 3 = calibration drift)

---

### Prof. Xin Yuan

**Affiliation:** School of Engineering, Westlake University, Hangzhou, China
**Website:** https://en.westlake.edu.cn/faculty/xin-yuan.html
**Lab:** Sensing and Computational Imaging (SCI) Lab
**Status in paper:** ✅ Already listed as co-author #2

**Education & Career:**
- Associate Professor, Westlake University (2021–present)
- Video Analysis and Coding Lead Researcher, Bell Labs (2015–2021)
- Postdoc, Duke University (2012–2015) — worked with David Brady
- PhD, The Hong Kong Polytechnic University (2012)

**Metrics:** ~15,600+ citations, 70+ journal papers, 60+ conference papers

**Awards:** National Excellent Young Scholar (overseas, 2021); Distinguished Young Scholar of Zhejiang Province (2022)

**Core Expertise:**
- Snapshot Compressive Imaging (SCI) — author of the definitive survey
- CASSI reconstruction algorithms (GAP-TV, DeSCI, PnP methods)
- CACTI video compressive imaging (EfficientSCI, deep unfolding)
- SPC / lensless compressive imaging (block-wise approach from Bell Labs)
- Deep learning for image reconstruction (end-to-end, plug-and-play, CNN-Transformer)

**Most Cited Works:**
1. "Snapshot Compressive Imaging: Theory, Algorithms and Applications" — IEEE SPM 2021 (with Brady & Katsaggelos)
2. "Rank Minimization for Snapshot Compressive Imaging" (DeSCI) — IEEE TPAMI 2019
3. "Plug-and-Play Algorithms for Large-Scale Snapshot Compressive Imaging" — CVPR 2020
4. "EfficientSCI: Densely Connected Network..." — CVPR 2023
5. "Block-wise Lensless Compressive Camera" — 2017

**Connection to PWM:**
- Central figure in snapshot compressive imaging (SCI)
- Unified CASSI, CACTI, SPC under one theoretical framework
- Developed GAP-TV (primary solver in PWM experiments)
- Developed EfficientSCI (used for CACTI validation)
- Worked with Brady at Duke (bridges both collaborators)
- His algorithms are reference implementations in the PWM codebase

---

## 9. Recommended Modalities to Add for Nature Submission

The paper already has 7 validated modalities across 4 carrier families — that's strong. The question is: **which additions fill the most critical gaps for Nature reviewers?**

### The 5 Modalities to Add (Prioritized)

#### #1. Ultrasound — MUST ADD

**Why this is non-negotiable for Nature:**
- The paper claims **5 carrier families** but only validates **4**. Acoustic is the missing one.
- A Nature reviewer will immediately ask: *"You claim universality across 5 carriers but show zero acoustic validation?"*
- Ultrasound is clinically ubiquitous (~200,000 scanners worldwide) — every Nature reader has had one

**Primitive chain:**
```
Source → C(h_PSF) → S(scan lines) → D(g, η_linear) → y
```
Uses: `C → S → D` (3 nodes, depth 3). All existing primitives.

**Gate 3 mismatch:** Speed-of-sound assumption (1540 m/s in soft tissue) vs. actual tissue-dependent variation (1450–1600 m/s). This directly causes geometric distortion and defocus — a textbook Gate 3 parameter.

**Effort:** Low. Publicly available datasets (PICMUS, Plane-Wave Imaging Challenge). Reuses existing primitives.

---

#### #2. Cryo-EM — MUST ADD

**Why this is critical for Nature:**
- Biology is Nature's **core audience**. The paper currently has zero biology modalities.
- Cryo-EM won the **2017 Nobel Prize in Chemistry** — Nature readers know it instantly
- CTF (Contrast Transfer Function) estimation is the rate-limiting calibration step in every cryo-EM workflow — this is **textbook Gate 3**
- Structural biologists will immediately understand *"operator mismatch limits your resolution"*

**Primitive chain:**
```
Source → P(electron wave) → M(specimen) → P(objective lens) → C(CTF) → D(DED)
```
Uses: `P → M → P → C → D` (5 nodes, depth 5). Adds the **coherent electron** sub-carrier.

**Gate 3 mismatch:** Defocus estimation error in the CTF. A 50nm defocus error at 300kV can limit resolution from 2Å to 3Å — the difference between a publishable and unpublishable structure.

**Effort:** Medium. Public datasets available (EMPIAR database). Shares `P` primitive with ptychography.

---

#### #3. CBCT (Cone-Beam CT) — STRONGLY RECOMMENDED

**Why important for Nature:**
- Extends CT from the simple parallel-beam case to the clinically dominant cone-beam geometry
- Critical for attracting **Jiang** (his deepest research area)
- Demonstrates the framework handles **scatter** as a Gate 3 mismatch — scatter correction is CBCT's #1 unsolved problem
- ~10,000 CBCT-equipped linacs in the US alone — massive clinical relevance

**Primitive chain:**
```
Source → Π(θ, cone) → C(scatter PSF) → D(flat-panel) → y
```
Uses: `Π → C → D` (3 nodes, depth 3). Extends existing CT template.

**Gate 3 mismatch:** Scatter kernel (patient-dependent, never perfectly known) + geometric calibration drift (source-detector alignment). Beam hardening uses `Λ` (Transform).

**Effort:** Low. Extends the validated CT template. Public CBCT datasets exist.

---

#### #4. Compressive Holography — STRONGLY RECOMMENDED

**Why important for Nature:**
- Critical for attracting **Brady** (he won the 2023 Leith Medal for inventing this)
- Demonstrates PWM handles **coherent 3D imaging** — a fundamentally different regime from current optical modalities (all incoherent 2D)
- The forward model is elegant: `P → M → P → D` — clean, compact, and visually distinct from CASSI/CACTI

**Primitive chain:**
```
Source → P(d₁,λ) → M(object) → P(d₂,λ) → D(|·|², η_sq) → hologram
```
Uses: `P → M → P → D` (4 nodes, depth 4). Tests the `P` (Propagate) primitive in a new context.

**Gate 3 mismatch:** Propagation distance error (defocus), reference beam angle error, wavelength calibration error, twin-image contamination.

**Effort:** Medium. Coherent propagation code shares infrastructure with ptychography. Public holographic datasets exist.

---

#### #5. Fluorescence Microscopy — RECOMMENDED

**Why important for Nature:**
- Second biology-relevant modality (alongside Cryo-EM). Covers life sciences from both structural and cellular perspectives
- Every biology lab in the world has a fluorescence microscope
- PSF estimation / deconvolution is the daily calibration challenge — pure Gate 3
- Super-resolution variants (STED, PALM, STORM) are Nobel Prize territory (2014 Chemistry)

**Primitive chain:**
```
Source → M(excitation) → Λ(fluorescence) → C(emission PSF) → D(PMT/camera) → y
```
Uses: `M → Λ → C → D` (4 nodes, depth 4). Tests `Λ` (Transform) for the fluorescence nonlinearity.

**Gate 3 mismatch:** PSF model error (theoretical Airy disk vs. actual aberrated PSF), sample-induced aberrations, photobleaching changing the effective `Λ`.

**Effort:** Medium. Public datasets from deconvolution challenges (e.g., ISBI benchmarks).

---

### What NOT to Add (Save for Later)

| Modality | Why not now |
|----------|-----------|
| Smartphone photography | Better as a Discussion paragraph + supplementary example, not a validated modality |
| PET | Medium effort, less unique than ultrasound for filling the acoustic gap. Save for Jiang's deeper involvement post-acceptance |
| Gigapixel camera | Less fundamental than holography for demonstrating PWM's range. Better as a post-publication extension |
| 4D-MRI | Variant of existing MRI, adds less novelty than a new carrier |
| SAR | Interesting but niche audience for Nature |
| JWST | Better as a Discussion mention than a validation |

---

### Before vs. After Comparison

| Dimension | Current Paper | After Adding 5 |
|-----------|--------------|----------------|
| Validated modalities | 7 | **12** |
| Carrier families validated | 4 of 5 (no acoustic) | **5 of 5** |
| Biology modalities | 0 | **2** (Cryo-EM + Fluorescence) |
| Clinical modalities | 2 (CT, MRI) | **4** (+ CBCT, Ultrasound) |
| Coherent 3D imaging | 0 | **1** (Holography) |
| Brady hook | CASSI only | CASSI + **Holography** |
| Jiang hook | CT only | CT + **CBCT** |
| Nature audience coverage | Physics/engineering | **+ Biology + Medicine** |
| Held-out closure test | 8 | **8+** (unchanged or expand) |
| Registered templates | 26 | **31+** |

---

### Implementation Priority Order

If time is limited, add in this order:

1. **Ultrasound** — Low effort, fills the acoustic carrier gap (a reviewer will flag this)
2. **CBCT** — Low effort, extends CT template, hooks Jiang
3. **Cryo-EM** — Medium effort, hooks Nature's biology audience (the single biggest impact on reviewer reception)
4. **Compressive Holography** — Medium effort, hooks Brady
5. **Fluorescence Microscopy** — Medium effort, deepens biology coverage

Adding just **#1 and #2** (both low effort) already significantly strengthens the paper. Adding all 5 makes it definitively Nature-level breadth: **12 modalities, 5 carriers, from hospital scanners to biology labs to coded aperture cameras**.
