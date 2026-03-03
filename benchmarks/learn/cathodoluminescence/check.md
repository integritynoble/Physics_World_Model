# Benchmark Review -- Cathodoluminescence (CL) Imaging

**URL:** https://pwm.platformai.org/benchmark/cathodoluminescence
**Review Date:** 2026-03-03
**Modality ID:** `cathodoluminescence`
**Category:** Scientific Instrumentation | **Carrier:** Electron | **Maturity:** M0

---

## 1. Platform Benchmark Page

Data extracted from the PWM platform page:

- **Title:** Cathodoluminescence (CL) Imaging -- Physics World Model
- **Canonical DAG:** M --> R --> D (Modulation --> Rotation --> Detector)
- **Forward Model Type:** nonlinear_operator (y = f(x) + n, where f is nonlinear)
- **Default Solver:** hyperspectral_unmixing
- **Physics Engine:** microscopy_psf (PSF convolution / deconvolution)
- **Image Shape:** x = [64, 64], y = [64, 64]

### Mismatch Parameters (from platform)

| Parameter | Symbol | Public Range | Unit | Physical Effect |
|-----------|--------|-------------|------|-----------------|
| Beam current drift | b_c | -1.0 to 2.0 | - | Temporal intensity variations |
| Collection efficiency variation | c_e | -4.0 to 8.0 | spatial | Detector response nonuniformity |
| Spectral calibration error | s_c | -0.4 to 0.8 | nm | Wavelength misalignment |
| Carbon contamination | c_c | -2.0 to 4.0 | - | Signal attenuation via specimen degradation |

### Evaluation Metrics (from platform)

Composite score = 0.4 x PSNR_norm + 0.4 x SSIM + 0.2 x (1 - ||y - Hx_hat|| / ||y||)

- **PSNR (40%):** Peak signal-to-noise ratio
- **SSIM (40%):** Structural similarity index
- **Consistency (20%):** Forward model fidelity

### Leaderboard (from platform)

| Rank | Method | Overall Score | PSNR (dB) | SSIM |
|------|--------|--------------|-----------|------|
| 1 | CalibFormer + gradient | 0.774 | 31.09 | 0.932 |
| 2 | ResNet-Calib + gradient | 0.653 | -- | -- |
| 3 | PnP-BM3D + gradient | 0.615 | -- | -- |
| 4 | Deconv + gradient | 0.575 | -- | -- |

Note: The previous check.md listed different method names (SpecTransformer + gradient,
EELS-Net + gradient, PnP-BM3D + gradient, PCA-Decomp + gradient). This discrepancy
should be investigated -- see Section 6 findings.

### Dataset Structure (from platform)

- **Public Tier (5 scenes):** Ground truth x_true, measurements y, ideal operator H, spec ranges, true mismatch parameters provided.
- **Dev Tier (5 scenes):** Blind evaluation -- only y, H, and spec ranges available.
- **Hidden Tier (5 scenes):** Fully blind server-side evaluation; algorithm containerized.
- **Source:** Protein Data Bank (PDB; Berman et al., NAR 2000).
- **Submission Format:** HDF5 (Public/Dev), containerized algorithm (Hidden).

---

## 2. Literature Review

### CL Imaging Fundamentals

Cathodoluminescence is the emission of photons when a material is excited by an
electron beam (typically inside an SEM or TEM). The technique provides nanoscale
spatial resolution combined with spectral information about the optical and
electronic properties of the sample. Key applications include semiconductors,
geological minerals, plasmonics, and defect characterization.

### Recent Advances (2024-2025)

**Non-perturbative CL Microscopy (Nanophotonics, 2025):**
Pan-sharpening techniques applied to CL microscopy to enable minimally-perturbative
high-spatial-resolution spectrum imaging of beam-sensitive materials. Addresses the
fundamental trade-off between electron dose (needed for signal-to-noise) and beam
damage in nanophotonic materials.
- Source: https://www.degruyterbrill.com/document/doi/10.1515/nanoph-2024-0724/html

**Synthetic Gain for Electron-Beam Spectroscopy (Nature Communications, 2025):**
Complex frequency waves created through causality-informed coherent superposition of
real-frequency waves induced by free electrons provide virtual gain to offset material
losses. Can retrieve resonance excitation buried underneath the zero-loss peak and
enhance hyperspectral imaging quality.
- Source: https://www.nature.com/articles/s41467-025-68189-z

**EELS Hyperspectral Images Unmixing Using Autoencoders (EPJ-AP, 2024):**
Deep learning autoencoders applied to spectral unmixing of electron-beam hyperspectral
data -- directly relevant to the PWM benchmark's default solver (hyperspectral_unmixing).
Demonstrates that learned representations can outperform classical PCA/NMF decomposition.
- Source: https://www.epjap.org/articles/epjap/full_html/2024/01/ap240025/ap240025.html

**CL Hyperspectral Imaging for Semiconductors (Microscopy & Microanalysis, 2024):**
Combining CL with other SEM modes for UV semiconductor characterization. Highlights
the importance of correlative approaches and noise-aware spectral processing.
- Source: https://academic.oup.com/mam/article/30/Supplement_1/ozae044.003/7719442

**Delmic + Digital Surf CL Workspace (2025):**
Commercial software for advanced CL data analysis powered by Mountains technology,
indicating growing industrial demand for standardized CL reconstruction tools.
- Source: https://www.digitalsurf.com/news/press-release-delmic-and-digital-surf-unveil-cl-workspace-software-packages-for-advanced-cathodoluminescence-data-analysis/

**Machine Learning for CL Classification (Talanta, 2022):**
Artificial neural networks achieve >97% accuracy for plastic classification using CL
spectra, demonstrating that ML/DL methods are effective on CL data.
- Source: https://www.sciencedirect.com/science/article/pii/S0039914022007810

### Gap Analysis

There is no large-scale public CL reconstruction benchmark in the literature.
Most CL reconstruction work is ad hoc, per-lab, and tied to specific instruments
(Delmic SPARC, Attolight, Gatan MonoCL). The PWM benchmark fills a genuine gap
by providing a standardized evaluation framework with controlled mismatch injection.

---

## 3. Local Dataset Status

**Local dataset directory:** `datasets/benchmark/cathodoluminescence/` -- DOES NOT EXIST.

No local data files were found. The benchmark config confirms:

- `dataset_id: ''` (empty)
- `dataset_url: ''` (empty)
- `fallback: generated` (uses `shepp_logan` synthetic generator)
- `data_source.priority: [experimental, synthetic_web, generated]`

The data source is entirely synthetic (generated). No experimental or web-sourced
CL datasets have been integrated. The platform page references PDB (Protein Data
Bank) as the source, which is unusual for CL data and may indicate that generic
structural phantoms are used rather than actual CL-specific data.

---

## 4. Local Configuration & Learn Materials

### Config Files

| File | Path | Status |
|------|------|--------|
| Base config | `benchmarks/configs/cathodoluminescence.yaml` | Present |
| Expanded config | `benchmarks/expanded_configs/cathodoluminescence_expanded.yaml` | Present |
| Modality doc | `docs/modality_benchmarks/cathodoluminescence.md` | Present |

### Learn Materials (benchmarks/learn/cathodoluminescence/)

| File | Size | Status |
|------|------|--------|
| README.md | 1,495 B | Present |
| 01_physics_fundamentals.md | 2,117 B | Present |
| 02_forward_model.md | 2,752 B | Present |
| 03_reconstruction_algorithms.md | 2,056 B | Present |
| 04_pwm_benchmark.md | 2,451 B | Present |
| 05_hands_on_tutorial.md | 3,573 B | Present |

### Expanded Config Details

- **Image sizes:** Small (128x128), Standard (256x256), Large (512x512)
- **Noise levels:** Clean (60 dB), Low (40 dB), Medium (30 dB), High (20 dB)
- **Mismatch levels:** M0 (nominal), M1 (single param), M2 (compound 3+), M3 (real), M4 (adversarial)
- **Total benchmark cases:** B1=12, B2=60, B3=60, B4=60, Grand Total=192

### Solvers

| Tier | Name | Module | GPU | Parameters |
|------|------|--------|-----|------------|
| traditional_cpu | Adjoint | pwm_core.recon.adjoint | No | 0 |
| best_quality | PnP-ADMM | pwm_core.recon.pnp_admm | Yes | 2M |

---

## 5. Benchmark Tier Verification (B1-B4)

From `docs/modality_benchmarks/cathodoluminescence.md`:

### B1: Design (Prompt + Original-Spec --> Spec)
- M0 through M4 defined with design template covering beam energy, spectral range,
  collection optics.
- 12 total cases.

### B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)
- Tests hyperspectral mapping under drift and beam damage.
- Mismatch injection at M0-M4 levels.
- 60 total cases.

### B3: System Identification (Dataset + Prompt --> Spec)
- Estimates spectral response, damage rate, drift from data.
- 60 total cases.

### B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)
- Corrects drift, damage model, spectral calibration.
- Expected rho: TBD (not yet defined).
- 60 total cases.

---

## 6. Findings & Recommendations

### ERRORS (must fix)

| ID | Severity | Finding |
|----|----------|---------|
| E1 | ERROR | **Wavelength range is 0-0 nm** in 01_physics_fundamentals.md (line: "0 -- 0 nm"). CL emission typically covers 200-1000 nm (UV to near-IR). This placeholder was never populated. |
| E2 | ERROR | **Base config x_shape/y_shape mismatch with expanded config.** Base config uses [64, 64] while expanded config defines Small=128x128, Standard=256x256, Large=512x512. The base config shape is smaller than even the "Small" expanded tier. |
| E3 | ERROR | **No local dataset exists.** `datasets/benchmark/cathodoluminescence/` is missing entirely. The fallback is synthetic `shepp_logan` phantoms, which have no spectral dimension and are not physically representative of CL data. |

### WARNINGS (should fix)

| ID | Severity | Finding |
|----|----------|---------|
| W1 | WARNING | **Leaderboard method name mismatch.** Platform shows CalibFormer/ResNet-Calib/PnP-BM3D/Deconv, but the previous check.md references SpecTransformer/EELS-Net/PnP-BM3D/PCA-Decomp. Only PnP-BM3D is consistent. Either the leaderboard was updated or there is an inconsistency between environments. |
| W2 | WARNING | **Platform mismatch ranges differ from local config.** Platform: beam_current_drift [-1.0, 2.0], local config: [0.0, 5.0]. Platform: collection_efficiency [-4.0, 8.0], local config: [0.0, 20.0]. Platform: spectral_cal [-0.4, 0.8], local config: [-2.0, 2.0]. Platform: carbon_contamination [-2.0, 4.0], local config: [0.0, 10.0]. The platform appears to show public-tier ranges while the config shows the full parameter space. This should be documented clearly. |
| W3 | WARNING | **PDB data source is unusual for CL.** The platform cites Protein Data Bank (Berman et al., NAR 2000) as the data source. PDB provides protein crystal structures, not CL measurements. If structural phantoms are being generated from PDB coordinates, this should be explicitly stated. |
| W4 | WARNING | **B4 expected rho is TBD.** The correction benchmark has no defined performance target, making it impossible to assess pass/fail. |
| W5 | WARNING | **Signal equation is generic electron microscopy CTF**, not CL-specific. The equation I(r) = |F^-1{CTF(q) . F{V(r)}}|^2 + noise describes coherent electron imaging (TEM phase contrast), not cathodoluminescence. CL involves incoherent photon emission from electron-beam excitation. A more appropriate model would be: I_CL(r,lambda) = integral[G(r,r',lambda) * S(r') * eta(r,lambda)] dr' + noise, where G is the generation volume, S is the source distribution, and eta is the collection efficiency. |
| W6 | WARNING | **Imaging chain elements are generic.** The forward model lists "Coded Mask" as a modulation element, but CL systems do not use coded masks. CL uses a focused electron beam as the excitation source (point-scanning) and a parabolic mirror or optical fiber for photon collection. |

### INFO (informational)

| ID | Severity | Finding |
|----|----------|---------|
| I1 | INFO | **192 total benchmark cases** across B1-B4 with 5 mismatch levels and 4 noise tiers. This is a reasonable evaluation matrix. |
| I2 | INFO | **Two solvers registered:** Adjoint (CPU baseline) and PnP-ADMM (GPU best quality). The literature suggests autoencoder-based spectral unmixing and transformer architectures are competitive -- these could be added as additional solver tiers. |
| I3 | INFO | **No competing public CL reconstruction benchmark exists.** This benchmark addresses a genuine gap in the community. |
| I4 | INFO | **Learn materials are complete** (all 5 tutorial files plus README present) but contain generic/templated content that should be specialized for CL physics. |

### Summary

| Severity | Count |
|----------|-------|
| ERROR | 3 |
| WARNING | 6 |
| INFO | 4 |

### Priority Actions

1. **[E1]** Populate the CL wavelength/energy range: typical CL covers 200-1000 nm (1.2-6.2 eV).
2. **[E3]** Source or generate CL-specific benchmark data. Options: (a) synthetic CL from known crystal structures using Monte Carlo electron-matter simulation (CASINO, DTSA-II), (b) experimental CL hyperspectral cubes from open repositories, (c) at minimum, spectral phantom data with realistic emission profiles rather than Shepp-Logan.
3. **[W5, W6]** Replace the generic CTF-based signal equation and coded-mask imaging chain with CL-appropriate physics: point-scanning excitation, incoherent photon emission, parabolic mirror collection, spectral dispersion.
4. **[E2]** Reconcile base config [64,64] with expanded config size tiers.
5. **[W2]** Document the relationship between platform public-tier ranges and full config ranges.
6. **[W4]** Define the B4 expected rho target for the correction benchmark.

---

*Comprehensive review of the cathodoluminescence PWM benchmark modality. Covers platform page, literature context, local dataset status, configuration verification, benchmark tier structure, and actionable findings.*

<!-- tags: cathodoluminescence, CL, electron-beam, hyperspectral, unmixing, SEM, benchmark, review, comprehensive -->