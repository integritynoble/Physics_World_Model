# PWM Per-Modality Benchmark Specifications

> **Scope**: Deep, actionable benchmark specifications (B1-B4) for all 97 imaging modalities.
> Each modality entry includes: example prompts, mismatch parameter ranges, True-Spec fields,
> correction targets, expected recovery metrics, and improvement suggestions.
>
> **Companion doc**: [`pwm_medical_physicist_targets.md`](pwm_medical_physicist_targets.md) for architecture overview.

---

## 0. Advisory: Modality Expansion & Benchmark Improvement Strategy

### 0.1 Should You Increase Imaging Modalities Beyond 97?

**Current state**: 97 modalities across 13 categories. 7 validated, 53 registered, 4 planned, 33 new (from expansion).

**Recommendation: Yes, but strategically. Target 120-130 modalities.**

Expand in these high-value gaps:

| Gap Area | Missing Modalities | Why Add | Priority |
|----------|--------------------|---------|----------|
| **Spectroscopy** | Raman imaging, IR spectroscopy, NMR imaging, mass spec imaging (SIMS, DESI) | Large user base in chemistry/pharma; unique R→D DAGs | HIGH |
| **Ultrafast imaging** | Streak camera, pump-probe, compressed ultrafast photography (CUP) | Growing field; temporal compression maps to Σ→D | HIGH |
| **Quantum imaging** | Ghost imaging, quantum illumination, entangled photon microscopy | Frontier physics; tests R primitive limits | MEDIUM |
| **Acoustic/seismic** | Full-waveform inversion (FWI), ocean acoustic tomography, medical HIFU monitoring | P→D family with complex propagation | MEDIUM |
| **Multi-modal fusion** | PET/CT, PET/MR, SPECT/CT, US/MRI, CT+fluorescence | Cross-modality Spec merging; tests B1 design complexity | HIGH |
| **Emerging medical** | Magnetic particle spectroscopy, electrical impedance spectroscopy, microwave imaging | New clinical tools; unique forward models | LOW |
| **Astronomy** | Coronagraphy, interferometric imaging (ALMA), lucky imaging | M→P→D with extreme contrast requirements | LOW |

**Do NOT expand** into modalities that are trivially isomorphic to existing ones (e.g., adding "confocal spinning-disk" when `confocal_livecell` covers it, or adding "dental CT" when `cbct` covers it). The Finite Primitive Basis theorem predicts saturation at ~10 canonical types regardless of modality count.

**Decision criterion**: Add a modality only if it introduces (a) a new DAG topology not covered by existing entries, (b) a new mismatch physics (e.g., quantum correlations), or (c) a large user community that would benefit from PWM benchmarks.

### 0.2 How to Improve Each Benchmark

**Systematic improvement methodology** — apply these 8 strategies per modality:

| # | Strategy | Applies to | How to Implement |
|---|----------|------------|------------------|
| 1 | **Real datasets over synthetic** | B2, B3, B4 | Replace Gaussian phantoms with published experimental data (Zenodo, TCIA, public repos). Every validated modality should have >=1 real dataset. |
| 2 | **Compound mismatch** | B2, B4 | Currently most benchmarks test single-parameter mismatch. Add compound scenarios: mask shift + gain drift + noise increase simultaneously. |
| 3 | **Adversarial mismatch (Red Team)** | B2, B3, B4 | Inject worst-case mismatch combinations found by optimization. Tests robustness beyond random perturbation. |
| 4 | **Cross-modality transfer** | B1 | Test whether a Spec designed for one modality (e.g., CASSI) can be adapted for a related modality (e.g., CACTI) via B1 prompt engineering. |
| 5 | **Time-varying mismatch** | B3, B4 | Real systems drift. Add time-series mismatch: parameters change across frames/slices (gain drift, thermal expansion, vibration). |
| 6 | **Uncertainty quantification** | B3, B4 | Require True-Spec comparison to include uncertainty bands: "estimated shift_x = 1.5 +/- 0.3 px" vs True-Spec shift_x = 1.47 px. |
| 7 | **Multi-scale evaluation** | B2, B4 | Evaluate reconstruction at multiple scales: global PSNR, ROI-specific SSIM, edge preservation (FSIM), spectral fidelity (SAM for hyperspectral). |
| 8 | **Feedback actionability scoring** | B2, B4 | Score feedback not just for correctness but for actionability: "increase photon budget 2x" is actionable; "improve system" is not. |

### 0.3 Benchmark Maturity Levels

Each modality's benchmarks progress through maturity levels:

| Level | B1 | B2 | B3 | B4 |
|-------|----|----|----|----|
| **M0** (Template) | Prompt template exists | Forward model template | DAG template | Correction template |
| **M1** (Synthetic) | Prompt tested on synthetic | Single-param mismatch, synthetic data | Synthetic True-Spec | Synthetic correction, single param |
| **M2** (Compound) | Multiple prompt variants | Compound mismatch, 3+ params | Compound identification | Compound correction, rho measured |
| **M3** (Real data) | Prompts grounded in real protocols | Real experimental data | Real True-Spec from calibration | Real data correction, rho >= 0.80 |
| **M4** (Adversarial) | Adversarial prompt attacks | Red Team mismatch injection | Adversarial identification | Adversarial correction + live feedback |

---

## 1. Microscopy (16 modalities)

---

### 1.1 Widefield Fluorescence Microscopy (`widefield`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Current maturity**: M3

#### B1: Design

**Example prompt**: "Design a widefield fluorescence microscope for GFP-labeled fixed cells: 60x oil objective, NA 1.4, emission 500-550 nm, pixel size 100 nm, FOV 80 um, sCMOS detector."

**Design parameters to specify in Spec**:
| Parameter | Range | Unit |
|-----------|-------|------|
| Objective NA | 0.4 - 1.49 | - |
| Magnification | 10x - 100x | - |
| PSF sigma (lateral) | 0.8 - 4.0 | pixels |
| Emission wavelength | 400 - 800 | nm |
| Pixel size | 50 - 260 | nm |
| Read noise | 1.0 - 10.0 | e- |
| FOV | 20 - 500 | um |

**Evaluation criteria**: (a) PSF matches Abbe diffraction limit for stated NA/wavelength, (b) pixel size satisfies Nyquist (< lambda/4NA), (c) photon budget sufficient for target SNR.

**Improvement suggestions**:
- Add depth-of-field constraint for thick samples
- Include illumination uniformity specification (< 5% variation across FOV)
- Specify chromatic correction requirements for multi-color imaging

#### B2: Forward + Reconstruct

**Mismatch parameters** (inject one or more):
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| PSF sigma | 2.0 | [1.2, 3.5] | px |
| Background level | 50 | [0, 200] | counts |
| Gain | 1.0 | [0.85, 1.15] | - |
| Flatfield non-uniformity | 0% | [0%, 15%] | peak-to-peak |
| Photobleaching rate | 0 | [0, 0.05] | per frame |

**Solvers**: Richardson-Lucy (50 iters), PnP-HQS (BM3D, 30 iters), Wiener filter.

**Expected Scenario I PSNR**: 30-38 dB (depends on photon budget).
**Expected Scenario II PSNR drop**: 1-5 dB (mild — PSF mismatch is gentle for widefield).

**Feedback template**: "PSF sigma mismatch: measured {x} px vs assumed {y} px. Reconstruction artifact: ring-shaped residuals in Fourier domain. Recommendation: re-measure PSF with sub-diffraction beads."

**Improvement suggestions**:
- Add compound mismatch: PSF + background + flatfield simultaneously
- Add depth-dependent PSF variation for 3D samples imaged in 2D

#### B3: System Identification

**True-Spec parameters** (hidden):
| Parameter | True Value Example | Contestant Knows |
|-----------|-------------------|-----------------|
| PSF sigma_x | 2.13 | Range [1.0, 4.0] |
| PSF sigma_y | 2.07 | Range [1.0, 4.0] |
| Background | 47.3 | Range [0, 200] |
| Read noise | 5.8 | Range [1, 15] |
| Gain | 1.03 | Range [0.8, 1.2] |

**Evaluation**: Parameter RMSE, PSF correlation with true PSF, noise model identification accuracy.

**Improvement suggestions**:
- Add asymmetric PSF (astigmatism) as identification target
- Include spatially-varying PSF across FOV

#### B4: Correction

**Correction targets**: Correct PSF model, background, flatfield.
**Expected rho**: >= 0.85 (widefield is forgiving).
**Expected PSNR gain**: +1 to +5 dB.

**Feedback quality targets**: Identify (a) PSF mismatch as dominant or noise as dominant, (b) recommend measured PSF or increased photon budget, (c) flag coverslip thickness error if astigmatism detected.

**Improvement suggestions**:
- Add time-varying mismatch: photobleaching changes effective signal across frames
- Test feedback actionability: "re-measure PSF with 170 um coverslip" vs "improve PSF"

---

### 1.2 Low-Dose Widefield (`widefield_lowdose`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Current maturity**: M1

#### B1: Design

**Example prompt**: "Design a live-cell widefield system minimizing phototoxicity: 40x water objective, NA 1.1, maximum 100 photons/pixel/frame, sCMOS with 1.0 e- read noise, 10 ms exposure."

**Design parameters**:
| Parameter | Range | Unit |
|-----------|-------|------|
| Photon budget | 10 - 500 | photons/px |
| Read noise | 0.5 - 5.0 | e- |
| Exposure time | 1 - 100 | ms |
| LED power fraction | 0.5 - 10 | % |
| Camera QE | 0.7 - 0.95 | - |

**Improvement suggestions**:
- Include phototoxicity model: cumulative dose vs cell viability
- Specify temporal resolution requirement (frames/s)

#### B2: Forward + Reconstruct

**Mismatch parameters**:
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Photon rate alpha | 100 | [10, 500] | photons/px |
| Read noise sigma | 5.0 | [1.0, 15.0] | e- |
| Background | 50 | [10, 200] | counts |
| Dark current | 0.1 | [0.01, 1.0] | e-/px/s |
| Hot pixel fraction | 0 | [0, 0.5%] | - |

**Solvers**: VST+BM3D, CARE, Noise2Void, PURE-LET.

**Expected Scenario I PSNR**: 18-25 dB (low photons).
**Expected Scenario II PSNR drop**: 3-10 dB (noise model mismatch is critical at low counts).

**Improvement suggestions**:
- Test mixed Poisson-Gaussian noise model vs pure Poisson assumption
- Add spatially-varying background (autofluorescence)
- Compound: low photons + hot pixels + background

#### B3: System Identification

**True-Spec parameters**:
| Parameter | True Value Example | Contestant Knows |
|-----------|-------------------|-----------------|
| Alpha (photon rate) | 87 | Range [10, 500] |
| Read noise | 4.2 | Range [1, 15] |
| Background | 63 | Range [10, 200] |
| Dark current | 0.15 | Range [0.01, 1.0] |

**Improvement suggestions**:
- Include camera sCMOS column-correlated noise pattern identification

#### B4: Correction

**Expected rho**: >= 0.70 (noise-limited; correction helps but ceiling is low).
**Expected PSNR gain**: +2 to +8 dB.

**Improvement suggestions**:
- Feedback should recommend "increase photon budget by Nx" with quantitative estimate
- Test whether noise model correction outperforms denoiser-only approach

---

### 1.3 Confocal Live-Cell (`confocal_livecell`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Current maturity**: M1

#### B1: Design

**Example prompt**: "Design confocal live-cell imaging: 63x oil, NA 1.4, pinhole 1 AU, scan speed 8 us/px, GFP channel, maximum 200 photons/px, 5 min timelapse at 30s intervals."

**Design parameters**:
| Parameter | Range | Unit |
|-----------|-------|------|
| Pinhole size | 0.5 - 3.0 | AU |
| Scan speed | 1 - 20 | us/px |
| Laser power | 0.1 - 10 | % |
| Z-step (if 3D) | 0.1 - 2.0 | um |
| Frame interval | 1 - 300 | s |

#### B2: Forward + Reconstruct

**Mismatch parameters**:
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| PSF sigma | 1.5 | [0.8, 3.0] | px |
| Drift rate | 0.1 | [0, 1.0] | px/frame |
| Bleaching rate | 0.01 | [0, 0.1] | per frame |
| Pinhole misalignment | 0 | [0, 0.5] | AU offset |

**Solvers**: RL + drift correction, CARE, deconvolution + registration.

**Improvement suggestions**:
- Add sample-induced aberration (refractive index mismatch in live cells)
- Compound: drift + bleaching + background increase over timelapse

#### B3: System Identification

**True-Spec parameters**: PSF, drift trajectory (px/frame, direction), bleaching curve, pinhole offset.

#### B4: Correction

**Expected rho**: >= 0.75.
**Feedback targets**: Recommend (a) drift correction method, (b) adaptive exposure to compensate bleaching, (c) pinhole realignment if offset detected.

---

### 1.4 Confocal 3D Z-Stack (`confocal_3d`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Current maturity**: M1

#### B1: Design

**Example prompt**: "Design 3D confocal z-stack: 100x oil, NA 1.45, 512x512x128 voxels, z-step 200 nm, depth 25 um, n_medium = 1.515."

**Design parameters**:
| Parameter | Range | Unit |
|-----------|-------|------|
| Axial PSF sigma | 1.5 - 6.0 | px |
| Lateral PSF sigma | 0.8 - 2.5 | px |
| Z-step | 0.1 - 1.0 | um |
| Depth range | 5 - 100 | um |
| Attenuation coefficient | 0.01 - 0.1 | per slice |
| Refractive index | 1.33 - 1.56 | - |

#### B2: Forward + Reconstruct

**Mismatch parameters**:
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Axial PSF sigma | 3.0 | [1.5, 6.0] | px |
| Refractive index | 1.515 | [1.33, 1.56] | - |
| Attenuation coeff | 0.03 | [0, 0.1] | per slice |
| Spherical aberration | 0 | [0, 0.5] | waves |

**Solvers**: 3D Richardson-Lucy, iterative constrained, multi-view fusion.

**Improvement suggestions**:
- Add depth-dependent PSF: PSF changes with distance from coverslip
- Include sample-induced scattering model for thick tissue

#### B3: True-Spec parameters

Refractive index, depth-dependent PSF model, attenuation profile, spherical aberration coefficient.

#### B4: Correction

**Expected rho**: >= 0.70 (depth-dependent aberration is hard to correct fully).
**Feedback**: "Spherical aberration = {x} waves at depth {z} um. Recommend: switch to silicone oil objective (n=1.406) for tissue imaging."

---

### 1.5 Structured Illumination Microscopy (`sim`)

**Canonical DAG**: M → C → D | **Carrier**: Photon | **Current maturity**: M2

#### B1: Design

**Example prompt**: "Design 2D-SIM system: 100x/1.49 TIRF objective, 3 orientations x 3 phases = 9 raw frames, pattern frequency 0.95x cutoff, modulation depth > 0.8, GFP channel."

**Design parameters**:
| Parameter | Range | Unit |
|-----------|-------|------|
| Pattern frequency | 0.5 - 1.0 | x cutoff |
| Number of orientations | 3 - 5 | - |
| Phases per orientation | 3 - 7 | - |
| Modulation depth | 0.3 - 1.0 | - |

#### B2: Forward + Reconstruct

**Mismatch parameters**:
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Pattern frequency | 0.1 | [0.05, 0.15] | cycles/px |
| Phase shifts | [0, 2pi/3, 4pi/3] | +/- 0.2 rad each | rad |
| Modulation depth | 0.8 | [0.3, 1.0] | - |
| Pattern orientation | [0, 60, 120] | +/- 3 deg each | deg |
| Bleaching per frame | 0 | [0, 0.05] | fraction |

**Solvers**: Wiener-SIM, HiFi-SIM, fairSIM, ML-SIM.

**Expected Scenario I PSNR**: 28-35 dB.
**Expected Scenario II PSNR drop**: 5-12 dB (SIM is very sensitive to pattern errors).

**Improvement suggestions**:
- Add nonlinear SIM (saturated) variant
- Test 3D-SIM with axial pattern estimation
- Compound: phase error + modulation fade + bleaching across orientations

#### B3: System Identification

**True-Spec parameters**: Pattern frequencies (3), phases (9), modulation depths (3), orientations (3), OTF attenuation.

**Key challenge**: Estimating 15+ parameters from 9 raw images. This is a hard identification problem.

#### B4: Correction

**Expected rho**: >= 0.80 (pattern correction is well-studied).
**Feedback**: "Phase shift error of {x} rad at orientation {i}. Modulation depth dropped to {y} at orientation {j} — possible polarization misalignment."

---

### 1.6 Light-Sheet (LSFM) (`lightsheet`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Current maturity**: M1

#### B1: Design

**Example prompt**: "Design light-sheet system for cleared mouse brain: 10x/0.6 detection, 5x/0.16 illumination, sheet thickness 5 um, FOV 1.5x1.5 mm, dual-sided illumination."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Sheet thickness | 5.0 | [2.0, 15.0] | um |
| Sheet tilt | 0 | [-3, 3] | deg |
| Stripe strength | 0.2 | [0, 0.8] | relative |
| Attenuation coeff | 0.02 | [0.005, 0.08] | per slice |
| Multi-view registration error | 0 | [0, 5] | px |

**Solvers**: Destripe + deconvolution, multi-view fusion (BigStitcher).

**Improvement suggestions**:
- Add scattering-induced stripe modeling
- Include tile stitching error for large-volume imaging

#### B3: True-Spec parameters

Sheet profile (Gaussian or Bessel), tilt angle, stripe pattern, attenuation profile, multi-view transformation matrices.

#### B4: Correction

**Expected rho**: >= 0.75.
**Feedback**: "Dominant artifact: horizontal stripes with period {p} px. Cause: absorbing structure in illumination path. Recommendation: enable dual-sided illumination or pivot illumination."

---

### 1.7 Fluorescence Lifetime (FLIM) (`flim`)

**Canonical DAG**: M → R → D | **Carrier**: Photon | **Current maturity**: M0

#### B1: Design

**Example prompt**: "Design FLIM system for FRET measurement: pulsed laser 405 nm, 80 MHz rep rate, TCSPC detector, 256 time bins, IRF width 80 ps, lifetime range 0.5-10 ns."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| IRF width | 80 | [40, 200] | ps |
| IRF shift | 0 | [-50, 50] | ps |
| Background (afterpulsing) | 0.01 | [0, 0.1] | relative |
| Pile-up fraction | 0 | [0, 0.05] | - |
| Time bin width | 50 | [40, 80] | ps |

**Solvers**: Phasor analysis, iterative reconvolution, Bayesian lifetime fitting.

**Improvement suggestions**:
- Add multi-exponential decay identification (2-3 lifetime components)
- Include FRET efficiency estimation benchmark

#### B3: True-Spec parameters

IRF shape, lifetimes (multi-component), fractional amplitudes, background, pile-up model.

#### B4: Correction

**Expected rho**: >= 0.70.
**Feedback**: "IRF width mismatch: estimated {x} ps vs nominal {y} ps. Pile-up detected at {z}%. Recommendation: reduce excitation power or apply pile-up correction."

---

### 1.8 Fourier Ptychographic Microscopy (`fpm`)

**Canonical DAG**: M → P → D | **Carrier**: Photon | **Current maturity**: M1

#### B1: Design

**Example prompt**: "Design FPM system: 4x/0.1NA objective, 15x15 LED array at 80 mm distance, LED spacing 4 mm, synthetic NA 0.5, 225 raw images."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| LED position x | array | +/- 0.5 mm each | mm |
| LED position y | array | +/- 0.5 mm each | mm |
| LED intensity | 1.0 | [0.5, 1.5] per LED | relative |
| Pupil aberration (Zernike) | 0 | [0, 0.3] waves per mode | waves |
| Defocus | 0 | [-5, 5] | um |

**Solvers**: Sequential phase retrieval, ePIE-FPM, Newton-FPM.

**Improvement suggestions**:
- Add vignetting model for off-axis LEDs
- Test robustness to LED failure (missing illuminations)

#### B3: True-Spec parameters

All LED positions (225x2), LED intensities (225), pupil aberration coefficients (Zernike Z4-Z11), defocus.

#### B4: Correction

**Expected rho**: >= 0.85 (LED position correction is very effective).
**Feedback**: "LED (7,3) displaced by {dx} mm east. Pupil shows coma = {z} waves. Recommendation: update LED calibration table; correct pupil in reconstruction."

---

### 1.9 Two-Photon / Multiphoton (`two_photon`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Current maturity**: M0

#### B1: Design

**Example prompt**: "Design two-photon system for in-vivo brain imaging: 25x/1.05 water objective, 920 nm excitation, GaAsP PMT, scan FOV 500x500 um, imaging depth 0-500 um."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Scattering coeff (mu_s) | 10 | [5, 30] | mm^-1 |
| PSF sigma (depth-dependent) | varies | scale [0.7, 1.5] | - |
| Excitation power attenuation | exp | coefficient [0.005, 0.02] | per um |
| Motion artifact | 0 | [0, 5] | um |

**Improvement suggestions**:
- Add adaptive optics correction benchmark
- Include in-vivo brain motion model (heartbeat, respiration)

#### B3: True-Spec parameters

Scattering coefficient profile, PSF vs depth curve, excitation attenuation, motion trajectory.

#### B4: Correction

**Expected rho**: >= 0.65 (scattering limits correction in deep tissue).

---

### 1.10 STED Microscopy (`sted`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Current maturity**: M0

#### B1: Design

**Example prompt**: "Design STED nanoscopy: 100x/1.4 oil, depletion beam 775 nm, saturation factor 30, effective resolution 40 nm, confocal reference channel."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Depletion beam alignment | 0 | [0, 30] | nm offset |
| Saturation factor | 30 | [10, 50] | - |
| Effective PSF FWHM | 40 | [30, 120] | nm |
| Background from incomplete depletion | 0 | [0, 0.2] | relative |

**Improvement suggestions**:
- Add 3D-STED benchmark
- Include bleaching rate model (STED causes more bleaching than confocal)

---

### 1.11 PALM/STORM Localization (`palm_storm`)

**Canonical DAG**: M → D | **Carrier**: Photon | **Current maturity**: M0

#### B1: Design

**Example prompt**: "Design STORM imaging: 100x/1.49 TIRF, Alexa Fluor 647, 10,000 frames, label density 500/um^2, target localization precision 20 nm."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Drift rate (x) | 0 | [0, 2] | nm/frame |
| Drift rate (y) | 0 | [0, 2] | nm/frame |
| Background photons | 20 | [5, 100] | per px per frame |
| Photon count per event | 1000 | [200, 5000] | photons |
| Camera pixel size | 100 | [90, 110] | nm |

**Solvers**: ThunderSTORM, DECODE, SMLM fitting.

**Improvement suggestions**:
- Add multi-emitter overlap benchmark
- Include 3D SMLM (astigmatic, biplane, or double-helix PSF)

#### B3: True-Spec parameters

Drift trajectory (x,y per frame), true label positions, photon rates per molecule, background map, camera calibration.

#### B4: Correction

**Expected rho**: >= 0.80 (drift correction is very effective).
**Feedback**: "Linear drift detected: {dx} nm/frame in x, {dy} nm/frame in y. Fiducial correction improves FRC resolution by {r} nm."

---

### 1.12 TIRF Microscopy (`tirf`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Incidence angle | 68 | [62, 75] | deg |
| Evanescent depth | 100 | [50, 300] | nm |
| Background (non-TIRF leak) | 0 | [0, 0.3] | relative |
| PSF sigma | 1.5 | [1.0, 3.0] | px |

**Improvement suggestions**:
- Add VA-TIRF (variable angle) benchmark
- Include objective-type vs prism-type TIRF comparison

---

### 1.13 Polarization Microscopy (`polarization`)

**Canonical DAG**: M → C → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Analyzer angle offset | 0 | [-5, 5] | deg |
| Retardance offset | 0 | [-10, 10] | nm |
| Polarizer extinction ratio | 1e-4 | [1e-5, 1e-3] | - |
| Detector gain imbalance (channels) | 1.0 | [0.9, 1.1] | per channel |

---

### 1.14 Expansion Microscopy (`expansion`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Expansion factor | 4.0 | [3.5, 4.5] | x |
| Local distortion | 0 | [0, 5%] | relative |
| Anisotropic expansion | 0 | [0, 3%] | x vs y |
| Labeling efficiency | 1.0 | [0.5, 1.0] | - |

**Improvement suggestions**:
- Add distortion field estimation benchmark (B3 should output spatial distortion map)
- Include multi-round expansion (iterative ExM)

---

### 1.15 MINFLUX Nanoscopy (`minflux`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Beam center position error | 0 | [0, 5] | nm |
| Beam pattern asymmetry | 0 | [0, 5%] | - |
| Photon count | 500 | [50, 2000] | photons |
| Background | 0.5 | [0, 5] | photons/px |

---

### 1.16 Image Scanning Microscopy (`ism`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Detector element offset | 0 | [-1, 1] | px |
| Magnification error | 0 | [-5%, 5%] | relative |
| Crosstalk between elements | 0 | [0, 10%] | - |

---

## 2. Compressive Imaging (4 modalities)

---

### 2.1 CASSI (`cassi`)

**Canonical DAG**: M → W → Sigma → D | **Carrier**: Photon | **Current maturity**: M3

#### B1: Design

**Example prompt**: "Design CASSI for surgical tissue spectral imaging: 28 bands at 450-650 nm, spatial resolution 256x256, binary random mask density 0.5, single-shot acquisition."

**Design parameters**:
| Parameter | Range | Unit |
|-----------|-------|------|
| Spectral bands | 8 - 64 | - |
| Spatial resolution | 128 - 1024 | px |
| Mask density | 0.3 - 0.7 | - |
| Dispersion per band | 1 - 4 | px |
| Spectral range | 400 - 1000 | nm |

#### B2: Forward + Reconstruct

**Mismatch parameters** (flagship-validated):
| Parameter | Nominal | Mismatch Range | True Example | Unit |
|-----------|---------|----------------|--------------|------|
| Mask shift dx | 0 | [-3.0, 3.0] | 1.47 | px |
| Mask shift dy | 0 | [-3.0, 3.0] | -0.23 | px |
| Mask rotation | 0 | [-2.0, 2.0] | 0.31 | deg |
| Dispersion slope a1 | 2.0 | [1.5, 2.5] | 2.01 | px/band |
| Dispersion offset alpha | 0 | [-0.5, 0.5] | 0.04 | px |
| Gain | 1.0 | [0.9, 1.1] | 1.02 | - |
| Read noise | 5.0 | [1.0, 15.0] | 5.1 | e- |

**Solvers**: GAP-TV, ADMM-TV, MST, CST, TSA-Net.

**Validated baseline** (flagship): GAP-TV +0.76 dB, rho = 0.85.

**Improvement suggestions**:
- Add compound mismatch: all 5 params simultaneously (validated at single-param level only)
- Add PSF mismatch (spectral PSF variation across bands)
- Red Team: adversarial worst-case combination of dx, dy, rotation
- Real dataset: TSA-Real (5 scenes, 660x660x28)

#### B3: System Identification

**True-Spec parameters**: dx, dy, rotation_deg, dispersion_slope, dispersion_offset, gain, read_noise, dark_current.

**Evaluation**: Parameter RMSE across all 5-7 mismatch params; must identify which Triad gate dominates (mismatch vs noise for CASSI is usually mismatch).

#### B4: Correction

**Expected rho**: >= 0.85 (validated).
**Expected PSNR gain**: +0.5 to +2.0 dB (more at high mismatch).

**Feedback template**: "Mask shift: dx = {est} px (true: {true}). Dispersion slope: a1 = {est} (true: {true}). Dominant gate: Operator Mismatch. Recommendation: (1) recalibrate mask alignment, expected gain +{x} dB; (2) re-measure dispersion curve, expected gain +{y} dB."

---

### 2.2 Single-Pixel Camera (`spc`)

**Canonical DAG**: M → Sigma → D | **Carrier**: Photon | **Current maturity**: M3

#### B1: Design

**Example prompt**: "Design SPC for NIR imaging: 64x64 resolution, Hadamard patterns, 25% sampling rate, DMD modulation, single InGaAs detector."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | True Example | Unit |
|-----------|---------|----------------|--------------|------|
| Gain drift alpha | 1.0 | [0.8, 1.2] | varies | - |
| Measurement noise sigma_y | 0.01 | [0, 0.1] | 0.03 | - |
| Pattern error (bit flips) | 0 | [0, 1%] | - | - |
| Timing jitter | 0 | [0, 5] | - | us |

**Validated baseline**: FISTA-TV +7.71 dB, rho = 0.86.

**Improvement suggestions**:
- Add time-varying gain drift (gain changes across measurement sequence)
- Test different pattern types: Hadamard vs Gaussian vs Fourier
- Include adaptive sampling (25% → 10% → 5%)

#### B3: True-Spec parameters

Gain curve (per measurement), noise level, pattern matrix perturbation.

#### B4: Correction

**Expected rho**: >= 0.86.

---

### 2.3 CACTI (`cacti`)

**Canonical DAG**: M → Sigma → D | **Carrier**: Photon | **Current maturity**: M3

#### B1: Design

**Example prompt**: "Design CACTI for high-speed video: 8 frames at 256x256, binary shifting mask, compression ratio 8:1."

#### B2: Mismatch parameters (flagship-validated)

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Spatial shift x | 0 | [-3, 3] | px |
| Spatial shift y | 0 | [-3, 3] | px |
| Rotation | 0 | [-2, 2] | deg |
| Temporal clock error | 0 | [-0.5, 0.5] | frame fraction |
| Gain | 1.0 | [0.9, 1.1] | - |
| Offset | 0 | [-5, 5] | counts |
| Mask density error | 0 | [-5%, 5%] | - |
| Frame-dependent gain | 1.0 | [0.9, 1.1] per frame | - |

**Validated baseline**: GAP-TV +10.21 dB, rho = 1.00.

**Improvement suggestions**:
- Test higher compression ratios (cr=16, cr=24)
- Add EfficientSCI real data (4 scenes, 512x512, cr=10)
- Compound: all 8 params simultaneously

#### B3: True-Spec parameters

All 8 mismatch params; mask replication pattern.

#### B4: Correction

**Expected rho**: >= 1.00 (flagship-validated; CACTI correction is very effective).
**Expected PSNR gain**: +5 to +10 dB.

---

### 2.4 Generic Matrix Sensing (`matrix`)

**Canonical DAG**: M → D | **Carrier**: varies | **Current maturity**: M1

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Matrix perturbation (Frobenius) | 0 | [0, 10%] of ||A|| | - |
| Condition number change | kappa | [kappa, 10*kappa] | - |
| Additive noise | 0 | [0, 0.1] | relative |

**Improvement suggestions**:
- Parameterize mismatch type: row deletion, column permutation, entry noise, rank deficiency

---

## 3. Medical Imaging (25 modalities)

---

### 3.1 X-ray CT (`ct`)

**Canonical DAG**: Pi → D | **Carrier**: X-ray | **Current maturity**: M3

#### B1: Design

**Example prompt**: "Design sparse-view CT for interventional guidance: 60 projections, fan-beam, 512 detector elements, 0.5mm resolution, ALARA dose."

**Design parameters**:
| Parameter | Range | Unit |
|-----------|-------|------|
| Number of projections | 30 - 1200 | - |
| Detector count | 256 - 2048 | - |
| Geometry | fan/parallel/cone | - |
| kVp | 60 - 140 | kV |
| Pixel size | 0.1 - 2.0 | mm |

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Center-of-rotation offset | 0 | [-5, 5] | px |
| Angular offset (global) | 0 | [-3, 3] | deg |
| Detector tilt | 0 | [-2, 2] | deg |
| Beam hardening coeff | 0 | [0, 0.05] | - |
| Ring artifact amplitude | 0 | [0, 50] | counts |

**Validated baseline**: FBP +10.68 dB, rho = 1.00.

**Datasets**: Walnut Micro-CT (Zenodo), Helsinki Tomography Challenge 2022, AAPM Low-Dose CT.

**Improvement suggestions**:
- Add metal artifact reduction benchmark
- Add limited-angle reconstruction (120 deg arc instead of 360)
- Include scatter correction benchmark for CBCT
- Compound: CoR offset + ring artifacts + beam hardening

#### B3: True-Spec parameters

CoR offset, angular errors per projection, detector pixel pitch, beam spectrum, ring artifact pattern.

#### B4: Correction

**Expected rho**: >= 1.00 (validated).
**Feedback**: "Center-of-rotation offset = {x} px. Ring artifact at detector elements {i, j, k}. Recommendation: recalibrate detector alignment, replace ring-artifact pixels."

---

### 3.2 MRI (`mri`)

**Canonical DAG**: M → F → S → D | **Carrier**: Spin/RF | **Current maturity**: M3

#### B1: Design

**Example prompt**: "Design accelerated brain MRI: 256x256, 8 coils, Cartesian trajectory, R=4 acceleration, 24 ACS lines, T2-weighted."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Coil sensitivity error | 0 | [0, 15%] per coil | relative |
| k-space trajectory deviation | 0 | [0, 2] | % |
| Off-resonance (B0 inhom.) | 0 | [-100, 100] | Hz |
| Acceleration factor | R=4 | [2, 8] | - |
| Noise level | varies | scale [0.5, 2.0] | - |

**Validated baseline**: SENSE +1.75 to +7.14 dB.

**Datasets**: fastMRI, Calgary-Campinas, M4Raw.

**Improvement suggestions**:
- Add non-Cartesian trajectory benchmark (radial, spiral)
- Include phase error estimation
- Add parallel imaging + compressed sensing combination
- Test at R=8 and R=16 (current baseline at R=2 and R=4)

#### B3: True-Spec parameters

Coil sensitivity maps (per coil), trajectory deviations, field map (B0), noise covariance matrix.

#### B4: Correction

**Expected rho**: >= 0.75.
**Feedback**: "Coil #3 sensitivity down 12% relative to nominal. B0 inhomogeneity: {x} Hz max in temporal lobe. Recommendation: re-estimate coil maps from ACS data, apply field-map correction."

---

### 3.3 X-ray Radiography (`xray_radiography`)

**Canonical DAG**: Pi → D | **Carrier**: X-ray | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Scatter fraction | 0 | [0, 0.4] | - |
| Beam hardening | none | polynomial order 2-4 | - |
| Detector lag | 0 | [0, 0.1] | fraction |
| Geometric magnification | 1.0 | [0.95, 1.05] | - |

---

### 3.4 Ultrasound B-mode (`ultrasound`)

**Canonical DAG**: P → D | **Carrier**: Acoustic | **Current maturity**: M1

#### B1: Design

**Example prompt**: "Design abdominal ultrasound: 128-element linear array, 3.5 MHz center frequency, focus depth 80 mm, imaging depth 150 mm."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Speed of sound | 1540 | [1450, 1600] | m/s |
| Phase aberration screen | 0 | [0, 50] | ns rms |
| Element sensitivity variation | 1.0 | [0.7, 1.3] per element | - |
| Frequency-dependent attenuation | 0.5 | [0.3, 0.8] | dB/cm/MHz |

**Solvers**: DAS, adaptive beamforming, synthetic aperture.

**Improvement suggestions**:
- Add aberration correction benchmark (tissue inhomogeneity)
- Include plane-wave ultrafast imaging mode
- Test shear-wave elastography coupling

#### B3: True-Spec parameters

Sound speed profile, aberration screen, element sensitivity map, attenuation profile.

#### B4: Correction

**Expected rho**: >= 0.70 (aberration correction is challenging in practice).

---

### 3.5 PET (`pet`)

**Canonical DAG**: Pi → D | **Carrier**: Gamma | **Current maturity**: M1

#### B1: Design

**Example prompt**: "Design whole-body PET/CT: BGO crystal ring, 4 mm resolution, TOF resolution 500 ps, 3-min per bed position, FDG tracer."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Attenuation map error | 0 | [0, 10%] | HU-to-LAC |
| Scatter fraction | 0.35 | [0.2, 0.5] | - |
| Randoms fraction | 0.2 | [0.1, 0.4] | - |
| Normalization error | 0 | [0, 5%] per detector | - |
| Dead time correction error | 0 | [0, 3%] | - |
| TOF timing offset | 0 | [-200, 200] | ps |

**Solvers**: MLEM, OSEM, PSF+TOF-OSEM.

**Improvement suggestions**:
- Add PET/MR attenuation challenge (MR-based attenuation is harder)
- Include motion correction benchmark
- Test at different count levels (clinical vs research dose)

#### B3: True-Spec parameters

Attenuation map, scatter sinogram, normalization table, dead time model, TOF calibration.

#### B4: Correction

**Expected rho**: >= 0.80.
**Feedback**: "Block #47 normalization deviates 5% from calibration. Scatter model underestimates by 8% in abdomen. Recommendation: re-run blank scan, update scatter kernel."

---

### 3.6 SPECT (`spect`)

**Canonical DAG**: Pi → D | **Carrier**: Gamma | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Collimator response error | 0 | [0, 20%] | FWHM |
| Center-of-rotation | 0 | [-3, 3] | px |
| Attenuation map error | 0 | [0, 15%] | relative |
| Scatter window error | 0 | [0, 10%] | keV |

---

### 3.7 Fluoroscopy (`fluoroscopy`)

**Canonical DAG**: Pi → D | **Carrier**: X-ray | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Temporal lag coefficient | 0 | [0, 0.15] | - |
| Geometric pincushion | 0 | [0, 3%] | - |
| Veiling glare | 0 | [0, 10%] | - |
| Frame rate mismatch | 30 | [15, 60] | fps |

---

### 3.8 Mammography (`mammography`)

**Canonical DAG**: Pi → D | **Carrier**: X-ray | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Heel effect profile | measured | +/- 10% | relative |
| Scatter-to-primary ratio | 0.4 | [0.2, 0.8] | - |
| Detector MTF variation | 1.0 | [0.8, 1.2] | at Nyquist |
| AEC error | 0 | [-20%, 20%] | exposure |

---

### 3.9 DEXA (`dexa`)

**Canonical DAG**: Pi → D | **Carrier**: X-ray | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Effective low energy | 40 | [35, 50] | keV |
| Effective high energy | 80 | [70, 100] | keV |
| Calibration polynomial coeff | nominal | +/- 5% | - |
| Fat-lean tissue assumption | 0 | [0, 20%] | fat fraction error |

---

### 3.10 CBCT (`cbct`)

**Canonical DAG**: Pi → D | **Carrier**: X-ray | **Current maturity**: M0

#### B1: Design

**Example prompt**: "Design CBCT for image-guided radiation therapy: half-fan, 360 projections over 200 deg, flat-panel 1024x768, 1 mm voxels."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Scatter fraction | 0.4 | [0.2, 0.7] | - |
| Cone-beam artifact amplitude | varies | scale [0.5, 2.0] | - |
| Truncation extent | 0 | [0, 20%] | FOV |
| Detector offset | 0 | [-5, 5] | px |
| Gantry flex | 0 | [0, 2] | mm |

**Improvement suggestions**:
- Add scatter correction benchmark using Monte Carlo ground truth
- Include patient motion during slow rotation
- Test limited-arc reconstruction (200 deg instead of 360)

#### B3: True-Spec parameters

Scatter distribution, geometric flex trajectory, truncation mask, ring artifact map.

#### B4: Correction

**Expected rho**: >= 0.80.
**Feedback**: "Scatter fraction = {x}. Cupping artifact detected. Gantry flex: max {y} mm at 180 deg. Recommendation: apply scatter kernel, correct flex model."

---

### 3.11 Angiography (`angiography`)

**Canonical DAG**: Pi → D | **Carrier**: X-ray | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Motion between mask and contrast | 0 | [0, 10] | px |
| Misregistration (rotation) | 0 | [-3, 3] | deg |
| Contrast timing offset | 0 | [-0.5, 0.5] | s |

---

### 3.12 Diffuse Optical Tomography (`dot`)

**Canonical DAG**: M → R,P,R → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Absorption coeff (mu_a) | 0.01 | [0.005, 0.05] | mm^-1 |
| Reduced scattering (mu_s') | 1.0 | [0.5, 2.0] | mm^-1 |
| Source-detector coupling | 1.0 | [0.5, 1.5] per fiber | - |
| Boundary condition model | extrapolated | varies | - |

---

### 3.13 Photoacoustic (`photoacoustic`)

**Canonical DAG**: M → P → D | **Carrier**: Acoustic | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Speed of sound | 1540 | [1400, 1600] | m/s |
| Acoustic attenuation | 0 | [0, 0.5] | dB/cm/MHz |
| Fluence model error | 0 | [0, 30%] | relative |
| Grueneisen parameter | 0.8 | [0.5, 1.2] | - |
| Transducer element sensitivity | 1.0 | [0.7, 1.3] per elem | - |

**Improvement suggestions**:
- Add 3D photoacoustic tomography benchmark
- Include limited-view reconstruction (partial aperture)
- Test spectral unmixing for multi-wavelength PA

---

### 3.14 OCT (`oct`)

**Canonical DAG**: P+P → Sigma → D | **Carrier**: Photon | **Current maturity**: M1

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Dispersion coeff (GDD) | 0 | [-100, 100] | fs^2 |
| Dispersion coeff (TOD) | 0 | [-50, 50] | fs^3 |
| Reference arm position | optimal | +/- 50 | um |
| K-linearization error | 0 | [0, 0.5%] | relative |
| Roll-off attenuation | measured | +/- 20% | - |

**Improvement suggestions**:
- Add OCT angiography benchmark
- Include speckle reduction benchmark

---

### 3.15 Functional MRI (`fmri`)

**Canonical DAG**: M → F → S → D | **Carrier**: Spin/RF | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| EPI geometric distortion | 0 | [0, 5] | px |
| Signal dropout fraction | 0 | [0, 15%] | of voxels |
| Motion parameters (6 DOF) | 0 | [0, 3] mm / [0, 3] deg | - |
| Physiological noise (cardiac, resp) | 0 | SNR contribution [0, 30%] | - |

---

### 3.16 MR Spectroscopy (`mrs`)

**Canonical DAG**: M → F → S → D | **Carrier**: Spin/RF | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Lineshape distortion | Lorentzian | Gaussian-Lorentzian mix | - |
| Eddy current phase | 0 | [0, 0.5] | rad |
| Residual water amplitude | 0 | [0, 100x metabolite] | relative |
| Baseline drift | 0 | polynomial order 0-3 | - |
| Chemical shift referencing | 0 | [-0.1, 0.1] | ppm |

---

### 3.17 Diffusion MRI (`diffusion_mri`)

**Canonical DAG**: M → F → S → D | **Carrier**: Spin/RF | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Gradient nonlinearity | 0 | [0, 5%] | relative |
| Eddy current distortion | 0 | [0, 3] | px |
| b-value error | 0 | [-5%, 5%] | relative |
| Gradient direction error | 0 | [0, 3] | deg |

---

### 3.18 Doppler Ultrasound (`doppler_ultrasound`)

**Canonical DAG**: P → D | **Carrier**: Acoustic | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Flow angle error | 60 | [0, 90] | deg |
| PRF aliasing threshold | correct | +/- 20% | - |
| Wall filter cutoff | 50 | [20, 200] | Hz |
| Clutter level | 0 | [0, -20] | dB below signal |

---

### 3.19 Shear-Wave Elastography (`elastography`)

**Canonical DAG**: P → D | **Carrier**: Acoustic | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Shear wave speed error | 0 | [-20%, 20%] | relative |
| Boundary reflection | none | [0, 30%] | amplitude |
| Dispersion (frequency-dependent) | none | [0, 20%] | at 200 Hz |

---

### 3.20 Fiber Bundle Endoscopy (`endoscopy`)

**Canonical DAG**: M → C → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Fiber transmission map error | 0 | [0, 15%] per fiber | - |
| Geometric distortion (barrel) | 0 | [0, 5%] | - |
| Inter-fiber cross-talk | 0 | [0, 10%] | - |
| Bending-induced attenuation | 0 | [0, 30%] | at max bend |

---

### 3.21 Fundus Camera (`fundus`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Lens aberration (Seidel) | 0 | [0, 0.5] | waves |
| Illumination non-uniformity | 0 | [0, 30%] | - |
| Pupil vignetting | 0 | [0, 20%] | at edge |

---

### 3.22 OCTA (`octa`)

**Canonical DAG**: P+P → Sigma → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Bulk motion amplitude | 0 | [0, 50] | um |
| Projection artifact strength | 0 | [0, 0.3] | relative |
| Shadow from large vessels | 0 | [0, 20%] | area |
| Interscan time variation | 5 | [3, 10] | ms |

---

### 3.23 Proton Therapy Imaging (`proton_therapy_img`)

**Canonical DAG**: Pi → D | **Carrier**: Proton | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Range uncertainty | 0 | [-3%, 3%] | - |
| Multiple Coulomb scattering model error | 0 | [0, 10%] | - |
| Energy straggling | nominal | +/- 15% | - |
| Nuclear interaction correction | 0 | [0, 5%] | - |

---

### 3.24 Brachytherapy Imaging (`brachytherapy_img`)

**Canonical DAG**: Pi → D | **Carrier**: Gamma/X-ray | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Source position error | 0 | [0, 3] | mm |
| Applicator geometry error | 0 | [0, 2] | mm |
| Tissue heterogeneity (vs TG-43) | homogeneous | real heterogeneity | - |

---

### 3.25 Portal Imaging / EPID (`portal_imaging`)

**Canonical DAG**: Pi → D | **Carrier**: MV X-ray | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Gantry sag | 0 | [0, 3] | mm |
| Detector arm flex | 0 | [0, 5] | mm |
| MLC leaf position error | 0 | [-1, 1] | mm |
| Backscatter correction | 0 | [0, 5%] | - |

---

## 4. Coherent Imaging (3 modalities)

---

### 4.1 Ptychography (`ptychography`)

**Canonical DAG**: M → P → D | **Carrier**: Electron/Photon | **Current maturity**: M3

#### B1: Design

**Example prompt**: "Design X-ray ptychography for nanostructure analysis: 500 eV soft X-ray, 64 scan positions with 70% overlap, Gaussian probe 30 nm diameter."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Probe position error (x) | 0 | [-5, 5] | px |
| Probe position error (y) | 0 | [-5, 5] | px |
| Defocus | 0 | [-50, 50] | nm |
| Aberration (astigmatism) | 0 | [0, 0.3] | waves |
| Probe intensity variation | 0 | [0, 10%] | - |
| Partial coherence | 1.0 | [0.7, 1.0] | - |

**Validated baseline**: ePIE +7.09 dB, rho = 1.00.

**Datasets**: 4D-STEM SrTiO3 [001] (128x128 scan, 300 kV, Zenodo 5113449).

**Improvement suggestions**:
- Add mixed-state ptychography benchmark (partial coherence)
- Include fly-scan (continuous scan) position refinement
- Test on thick samples requiring multi-slice ptychography

#### B3: True-Spec parameters

All probe positions (64x2), probe function, aberration coefficients, defocus, coherence function.

#### B4: Correction

**Expected rho**: >= 1.00 (validated).
**Feedback**: "Position error: probe (23) displaced by {dx} nm. Global defocus = {z} nm. Recommendation: refine positions with annealing schedule, update defocus in reconstruction."

---

### 4.2 Holography (`holography`)

**Canonical DAG**: P → D | **Carrier**: Photon | **Current maturity**: M1

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Reference beam angle error | 0 | [-1, 1] | deg |
| Carrier frequency error | 0 | [-5%, 5%] | - |
| Phase offset | 0 | [0, 2*pi] | rad |
| Vibration amplitude | 0 | [0, lambda/10] | - |
| Detector defocus | 0 | [-100, 100] | um |

**Improvement suggestions**:
- Add inline holography benchmark (twin image removal)
- Include phase-shifting holography with calibration errors
- Test wavelength-scanning for tomographic reconstruction

---

### 4.3 Phase Retrieval / CDI (`phase_retrieval`)

**Canonical DAG**: P → D | **Carrier**: Photon/Electron | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Support mask error | 0 | [0, 10%] area | - |
| Oversampling ratio | 2.0 | [1.5, 4.0] | - |
| Partial coherence | 1.0 | [0.7, 1.0] | - |
| Missing center pixels | 0 | [0, 10] | px radius |
| Detector saturation | none | [0, 5%] of pixels | - |

---

## 5. Computational Photography (2 modalities)

---

### 5.1 Lensless / Diffuser Camera (`lensless`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Current maturity**: M3

#### B1: Design

**Example prompt**: "Design diffuser-camera for microscopy: 256x256 resolution, random phase diffuser, sensor-to-diffuser distance 1 mm, monochromatic 520 nm."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| PSF lateral shift | 0 | [-5, 5] | px |
| PSF scale drift | 1.0 | [0.9, 1.1] | - |
| Defocus offset | 0 | [-50, 50] | um |
| PSF rotation | 0 | [-2, 2] | deg |
| Background | 0 | [0, 0.05] | relative |

**Validated baseline**: ADMM +3.55 dB, rho = 0.78.

**Improvement suggestions**:
- Add multi-depth reconstruction benchmark
- Test RGB PSF variation (chromatic aberration in diffuser)
- Real dataset: DiffuserCam (Waller Lab)

#### B3: True-Spec parameters

PSF function, shift, scale, defocus, rotation, background.

#### B4: Correction

**Expected rho**: >= 0.78 (validated).

---

### 5.2 Panorama Multi-Focus (`panorama`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Focus distance error | 0 | [-10%, 10%] per plane | - |
| Registration error | 0 | [0, 3] | px |
| Aperture variation | f/4 | [f/2, f/8] | - |

---

## 6. Computational Optics (2 modalities)

---

### 6.1 Light Field Imaging (`light_field`)

**Canonical DAG**: C → S → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Microlens pitch error | 0 | [-2%, 2%] | - |
| Rotation of microlens array | 0 | [-1, 1] | deg |
| F-number error | f/2 | [f/1.4, f/4] | - |
| Vignetting at edges | 0 | [0, 30%] | - |

---

### 6.2 Integral Photography (`integral`)

**Canonical DAG**: C → S → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Lens array position error | 0 | [0, 0.5] | mm |
| Distortion per lenslet | 0 | [0, 3%] | - |
| Fill factor | 1.0 | [0.8, 1.0] | - |

---

## 7. Neural Rendering (2 modalities)

---

### 7.1 NeRF (`nerf`)

**Canonical DAG**: M → P → D | **Carrier**: Photon | **Current maturity**: M1

#### B1: Design

**Example prompt**: "Design NeRF capture for indoor scene: 100 views on hemisphere, DSLR camera, focal length 35 mm, scene bounds 2-6 m."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Camera pose error (translation) | 0 | [0, 0.05] | scene units |
| Camera pose error (rotation) | 0 | [0, 3] | deg |
| Focal length error | 0 | [-5%, 5%] | - |
| Lens distortion (k1, k2) | 0 | [-0.2, 0.2] | - |
| Exposure variation | 0 | [0, 0.5] | stops |

**Solvers**: NeRF, Instant-NGP, Nerfacto, TensoRF.

**Improvement suggestions**:
- Add few-view NeRF benchmark (3-10 views)
- Include appearance variation (lighting changes between views)
- Test in-the-wild captures with transient objects

#### B3: True-Spec parameters

Camera poses (extrinsics per view), intrinsics (fx, fy, cx, cy, distortion), exposure per view, scene bounds.

#### B4: Correction

**Expected rho**: >= 0.80.
**Feedback**: "View #17 pose error: {dx} scene units translation. Global focal length bias: {f}%. Recommendation: run COLMAP SfM refinement, then retrain."

---

### 7.2 3D Gaussian Splatting (`gaussian_splatting`)

**Canonical DAG**: M → P → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| SfM point cloud noise | 0 | [0, 0.1] | scene units |
| Initialization density | 100k | [10k, 1M] | points |
| Camera pose error | 0 | [0, 0.03] | scene units |
| Missing views | 0 | [0, 20%] | of total |

---

## 8. Electron Microscopy (8 modalities)

---

### 8.1 SEM (`sem`)

**Canonical DAG**: C → D | **Carrier**: Electron | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Astigmatism (x, y) | 0 | [0, 50] | nm |
| Working distance error | 0 | [-0.5, 0.5] | mm |
| Drift rate | 0 | [0, 1] | nm/s |
| Charging level | 0 | [0, 500] | V |
| Beam energy | 5 | [1, 30] | keV |

---

### 8.2 TEM (`tem`)

**Canonical DAG**: C → D | **Carrier**: Electron | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Defocus | 0 | [-1000, 1000] | nm |
| Spherical aberration (Cs) | 1.2 | [0.5, 2.5] | mm |
| Astigmatism | 0 | [0, 100] | nm |
| Beam tilt | 0 | [0, 2] | mrad |
| Energy spread | 0.7 | [0.3, 1.5] | eV |

---

### 8.3 Electron Tomography (`electron_tomography`)

**Canonical DAG**: Pi → D | **Carrier**: Electron | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Tilt axis offset | 0 | [-3, 3] | px |
| Magnification variation | 0 | [-2%, 2%] per tilt | - |
| Missing wedge angle | 30 | [20, 50] | deg |
| Sample shrinkage | 0 | [0, 10%] | during acquisition |

---

### 8.4 STEM (`stem`)

**Canonical DAG**: S → D | **Carrier**: Electron | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Scan distortion (flyback) | 0 | [0, 3] | px |
| Scan rotation | 0 | [-2, 2] | deg |
| Probe aberration (defocus) | 0 | [-50, 50] | nm |
| Detector inner/outer angle | nominal | +/- 10% | mrad |

---

### 8.5 4D-STEM Diffraction (`electron_diffraction`)

**Canonical DAG**: M → P → D | **Carrier**: Electron | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Camera length error | 0 | [-5%, 5%] | - |
| Beam center (x, y) | center | +/- 5 | px |
| Diffraction pattern rotation | 0 | [-3, 3] | deg |
| Descan error | 0 | [0, 2] | px |

---

### 8.6 EBSD (`ebsd`)

**Canonical DAG**: R → D | **Carrier**: Electron | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Pattern center (x, y, z) | calibrated | +/- 2% each | - |
| Detector tilt | 0 | [-1, 1] | deg |
| Sample tilt | 70 | [68, 72] | deg |

---

### 8.7 EELS (`eels`)

**Canonical DAG**: S → D | **Carrier**: Electron | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Energy drift | 0 | [-2, 2] | eV |
| Gain instability | 0 | [0, 5%] per spectrum | - |
| Channel-to-channel gain variation | 0 | [0, 3%] | - |
| Dark current pattern | 0 | [0, 50] | counts |

---

### 8.8 Electron Holography (`electron_holography`)

**Canonical DAG**: P → D | **Carrier**: Electron | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Biprism voltage drift | 0 | [-2%, 2%] | - |
| Fringe spacing variation | 0 | [-3%, 3%] | - |
| Fringe rotation | 0 | [-1, 1] | deg |
| Fresnel fringe contamination | 0 | [0, 10%] | of FOV |

---

## 9. Depth Imaging (3 modalities)

---

### 9.1 Time-of-Flight Camera (`tof_camera`)

**Canonical DAG**: P → D | **Carrier**: Photon/IR | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Multi-path interference | none | [0, 30%] amplitude | - |
| Phase wrap count | correct | +/- 1 | wraps |
| Integration time variation | 1.0 | [0.5, 2.0] | ms |
| Temperature-dependent offset | 0 | [-5, 5] | cm |

---

### 9.2 LiDAR (`lidar`)

**Canonical DAG**: P → S → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Timing jitter | 0 | [0, 0.5] | ns |
| Angular encoder error | 0 | [-0.1, 0.1] | deg |
| Intensity calibration | 1.0 | [0.8, 1.2] | - |
| Multi-return ambiguity | none | [0, 5%] of points | - |

---

### 9.3 Structured-Light 3D (`structured_light`)

**Canonical DAG**: M → C → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Gamma nonlinearity | 1.0 | [1.5, 2.5] | - |
| Projector-camera extrinsics error | 0 | [0, 1] | mm / deg |
| Defocus-induced fringe blur | 0 | [0, 3] | px sigma |
| Ambient light contamination | 0 | [0, 20%] | of fringe contrast |

---

## 10. Remote Sensing (8 modalities)

---

### 10.1 SAR (`sar`)

**Canonical DAG**: F → D | **Carrier**: RF | **Current maturity**: M0

#### B1: Design

**Example prompt**: "Design stripmap SAR: X-band (9.65 GHz), 150 MHz bandwidth, PRF 3000 Hz, look angle 30 deg, 3m azimuth resolution."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Platform velocity error | 0 | [-1%, 1%] | relative |
| Motion compensation residual | 0 | [0, pi/4] | rad phase |
| Autofocus error | 0 | polynomial order 2-4 | - |
| Cross-range sidelobe | -13 | [-20, -8] | dB |

**Improvement suggestions**:
- Add InSAR (interferometric) benchmark with phase unwrapping
- Include change detection benchmark (multi-temporal)

---

### 10.2 Sonar (`sonar`)

**Canonical DAG**: P → D | **Carrier**: Acoustic | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Sound speed profile error | 0 | [-2%, 2%] | - |
| Multipath structure | none | 1-3 paths | - |
| Bottom reverberation | 0 | [0, -10] dB | - |
| Array element failure | 0 | [0, 5%] | of elements |

---

### 10.3 Hyperspectral Remote Sensing (`hyperspectral_remote`)

**Canonical DAG**: M → W → Sigma → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Smile distortion | 0 | [0, 2] | px |
| Keystone distortion | 0 | [0, 2] | px |
| Atmospheric model error | 0 | [0, 10%] | transmittance |
| Spectral response shift | 0 | [-2, 2] | nm per band |
| Striping (detector non-uniformity) | 0 | [0, 5%] | - |

**Improvement suggestions**:
- Add spectral unmixing benchmark (endmember extraction + abundance estimation)
- Include target detection benchmark (anomaly detection)

---

### 10.4 Multispectral Satellite (`multispectral_sat`)

**Canonical DAG**: M → Sigma → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Band-to-band registration | 0 | [0, 2] | px |
| MTF difference (pan vs MS) | measured | +/- 10% | - |
| Atmospheric path radiance | 0 | [0, 15%] | - |
| Cloud/shadow contamination | 0 | [0, 10%] | of pixels |

---

### 10.5 Ground-Penetrating Radar (`gpr`)

**Canonical DAG**: P → D | **Carrier**: RF | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Dielectric permittivity error | 0 | [-20%, 20%] | - |
| Clutter level | 0 | [0, -10] dB | relative to signal |
| Antenna coupling | measured | +/- 20% | - |
| Surface bounce removal | perfect | [80%, 100%] removal | - |

---

### 10.6 Weather Radar (`weather_radar`)

**Canonical DAG**: P → R → D | **Carrier**: RF | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Ground clutter power | -30 | [-40, -15] | dBZ |
| Path attenuation (heavy rain) | 0 | [0, 10] | dB/km at C-band |
| ZDR calibration offset | 0 | [-0.3, 0.3] | dB |
| Velocity aliasing (Nyquist) | correct | dealiasing needed | - |

---

### 10.7 Radio Interferometry (`radio_interferometry`)

**Canonical DAG**: F → S → D | **Carrier**: RF | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Baseline error | 0 | [0, 1] | cm |
| Atmospheric phase | 0 | [0, 1] | rad rms |
| Bandpass calibration | 1.0 | [0.9, 1.1] per channel | - |
| RFI contamination | 0 | [0, 5%] | of channels |

---

### 10.8 Passive Microwave Radiometry (`passive_microwave`)

**Canonical DAG**: Sigma → D | **Carrier**: RF | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Antenna pattern error | 0 | [0, 5%] | main beam |
| Gain calibration drift | 0 | [-1%, 1%] | per orbit |
| Cross-polarization leakage | 0 | [0, -20] dB | - |
| Thermal reference error | 0 | [-0.5, 0.5] | K |

---

## 11. Industrial Inspection (8 modalities)

---

### 11.1 Industrial CT (`industrial_ct`)

**Canonical DAG**: Pi → D | **Carrier**: X-ray | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Center offset | 0 | [-5, 5] | px |
| Scatter fraction | 0.3 | [0.1, 0.6] | - |
| Beam hardening | none | polynomial order 2-3 | - |
| Ring artifact sources | 0 | [0, 5] | detector elements |
| Magnification error | 0 | [-1%, 1%] | - |

**Improvement suggestions**:
- Add multi-material beam hardening correction (metal + plastic)
- Include dimensional metrology benchmark (measure feature sizes)

---

### 11.2 X-ray NDT (`xray_ndt`)

**Canonical DAG**: Pi → D | **Carrier**: X-ray | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Source-detector distance error | 0 | [-5%, 5%] | - |
| Geometric unsharpness | 0 | [0, 1] | mm |
| Scatter buildup factor | 1.0 | [1.0, 3.0] | - |
| Contrast sensitivity | measured | +/- 20% | - |

---

### 11.3 Ultrasonic Phased Array (`ultrasonic_phased_array`)

**Canonical DAG**: P → D | **Carrier**: Acoustic | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Velocity error | 0 | [-3%, 3%] | - |
| Coupling variation | 1.0 | [0.5, 1.5] per element | - |
| Element sensitivity | 1.0 | [0.7, 1.3] per element | - |
| Wedge angle error | 0 | [-2, 2] | deg |

---

### 11.4 Eddy Current (`eddy_current`)

**Canonical DAG**: F → D | **Carrier**: EM | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Lift-off variation | 0 | [0, 0.5] | mm |
| Conductivity error | 0 | [-10%, 10%] | - |
| Probe tilt | 0 | [-5, 5] | deg |
| Frequency response | nominal | +/- 10% | - |

---

### 11.5 Active Thermography (`active_thermography`)

**Canonical DAG**: P → D | **Carrier**: IR photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Emissivity map error | 0 | [0, 15%] | - |
| Heating uniformity | 1.0 | [0.7, 1.3] | spatial |
| Ambient temperature drift | 0 | [-2, 2] | deg C |
| Camera NETD error | 0 | [-20%, 20%] | - |

---

### 11.6 Terahertz Imaging (`terahertz`)

**Canonical DAG**: P → D | **Carrier**: THz photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Water vapor absorption | 0 | [0, 5] | dB at 1 THz |
| Etalon artifact period | none | [0, 100] | GHz |
| Refractive index error | 0 | [-5%, 5%] | - |
| Beam alignment | 0 | [0, 0.5] | mm |

---

### 11.7 Machine Vision / AOI (`machine_vision`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Illumination non-uniformity | 0 | [0, 20%] | - |
| Lens MTF at Nyquist | 0.5 | [0.2, 0.8] | - |
| Lens distortion | 0 | [0, 3%] | barrel/pincushion |
| Focus drift | 0 | [-50, 50] | um |

---

### 11.8 XRF Imaging (`xrf_imaging`)

**Canonical DAG**: M → R → D | **Carrier**: X-ray | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Matrix effect (absorption/enhancement) | 0 | [0, 20%] | - |
| Self-absorption correction | 0 | [0, 30%] | for heavy elements |
| Dead time correction | 0 | [0, 10%] | at high count rate |
| Pile-up fraction | 0 | [0, 5%] | - |
| Element overlap (spectral) | 0 | [0, 3] elements | - |

---

## 12. Scientific Instrumentation (8 modalities)

---

### 12.1 X-ray Crystallography (`xray_crystallography`)

**Canonical DAG**: F → S → D | **Carrier**: X-ray | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Absorption correction error | 0 | [0, 10%] | - |
| Radiation damage fraction | 0 | [0, 20%] per dataset | - |
| Crystal mosaicity error | 0 | [-50%, 50%] | relative |
| Detector distance error | 0 | [-1%, 1%] | - |
| Beam center error | 0 | [0, 2] | px |

---

### 12.2 SAXS (`saxs`)

**Canonical DAG**: R → D | **Carrier**: X-ray | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Beam divergence | 0.1 | [0.05, 0.5] | mrad |
| Parasitic scattering | 0 | [0, 20%] of signal | - |
| Background subtraction error | 0 | [-5%, 5%] | - |
| Sample thickness error | 0 | [-10%, 10%] | - |

---

### 12.3 MALDI Mass Spec Imaging (`maldi_msi`)

**Canonical DAG**: S → D | **Carrier**: Ion | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Mass calibration drift | 0 | [-10, 10] | ppm |
| Ion suppression variation | 0 | [0, 50%] | spatial |
| Matrix crystal inhomogeneity | 0 | [0, 30%] | relative |
| Baseline intensity variation | 0 | [0, 20%] | - |

---

### 12.4 Atom Probe Tomography (`atom_probe`)

**Canonical DAG**: S → D | **Carrier**: Ion | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Trajectory aberration | 0 | [0, 10%] | position error |
| Local magnification effect | 0 | [0, 20%] | - |
| Detection efficiency | 0.57 | [0.4, 0.8] | - |
| Mass resolution (FWHM) | 500 | [200, 2000] | M/dM |

---

### 12.5 Cryo-EM Single Particle (`cryo_em`)

**Canonical DAG**: C → D | **Carrier**: Electron | **Current maturity**: M0

#### B1: Design

**Example prompt**: "Design cryo-EM data collection: 300 kV, defocus range -1 to -3 um, pixel size 1.0 A, total dose 50 e-/A^2, ice thickness 30-50 nm."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Defocus per micrograph | -2.0 | [-0.5, -5.0] | um |
| Spherical aberration (Cs) | 2.7 | [2.0, 3.5] | mm |
| Beam tilt | 0 | [0, 1] | mrad |
| Ice thickness | 40 | [20, 100] | nm |
| Astigmatism | 0 | [0, 200] | nm |
| Phase plate offset | 0 | [0, pi/4] | rad |

**Improvement suggestions**:
- Add Ewald sphere curvature correction benchmark
- Include preferred orientation bias detection and correction
- Test at different particle sizes (100 kDa vs 1 MDa)

#### B3: True-Spec parameters

Per-micrograph CTF parameters (defocus, astigmatism, phase shift), beam tilt vector, ice thickness map.

#### B4: Correction

**Expected rho**: >= 0.85 (CTF correction is well-established but compound errors degrade it).
**Feedback**: "Micrograph #47: defocus = {x} um, astigmatism = {y} nm at {theta} deg. Beam tilt = {z} mrad. Recommendation: re-process with per-micrograph CTF; discard micrographs with ice > 80 nm."

---

### 12.6 Neutron Tomography (`neutron_tomo`)

**Canonical DAG**: Pi → D | **Carrier**: Neutron | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Beam spectrum error | white | +/- 10% | energy distribution |
| Scattering correction | 0 | [0, 15%] | of signal |
| Gamma contamination | 0 | [0, 5%] | of signal |
| Rotation axis offset | 0 | [-3, 3] | px |

---

### 12.7 Proton Radiography (`proton_radiography`)

**Canonical DAG**: Pi → D | **Carrier**: Proton | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| MCS model error | 0 | [0, 15%] | scattering angle rms |
| Energy loss model error | 0 | [-5%, 5%] | - |
| Detector alignment | 0 | [0, 1] | mm |
| Nuclear interaction correction | 0 | [0, 5%] | - |

---

### 12.8 Muon Tomography (`muon_tomo`)

**Canonical DAG**: Pi → D | **Carrier**: Muon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Angular resolution | 5 | [3, 15] | mrad |
| Detector layer alignment | 0 | [0, 1] | mm |
| Track fitting efficiency | 0.95 | [0.8, 1.0] | - |
| Integration time adequacy | sufficient | [50%, 200%] of nominal | - |

---

## 13. Broader Experimental Science (8 modalities)

---

### 13.1 Adaptive Optics (`adaptive_optics`)

**Canonical DAG**: M → C → D | **Carrier**: Photon | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Residual wavefront error | 0 | [0, lambda/4] | rms |
| DM actuator hysteresis | 0 | [0, 5%] | - |
| Fried parameter (r0) | 20 | [5, 30] | cm |
| Wind speed (frozen flow) | 10 | [5, 30] | m/s |
| Anisoplanatism angle | 0 | [0, 10] | arcsec |

---

### 13.2 Seismic Tomography (`seismic_tomo`)

**Canonical DAG**: P → D | **Carrier**: Seismic | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Velocity model error | 0 | [-10%, 10%] | - |
| Source location error | 0 | [0, 5] | km |
| Station timing error | 0 | [-0.1, 0.1] | s |
| Ray-bending model order | 1 | [1, 3] | - |

---

### 13.3 Gravitational Wave Detection (`gravitational_wave`)

**Canonical DAG**: P → Sigma → D | **Carrier**: Gravitational | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Calibration error (amplitude) | 0 | [-5%, 5%] | - |
| Calibration error (phase) | 0 | [-5, 5] | deg |
| Noise PSD estimation error | 0 | [-10%, 10%] | - |
| Glitch contamination rate | 0 | [0, 1] | per 100s |

---

### 13.4 Particle Calorimetry (`particle_calorimetry`)

**Canonical DAG**: R → Sigma → D | **Carrier**: Particle | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Inter-calibration error | 0 | [0, 3%] per cell | - |
| Non-linearity | 0 | [0, 5%] at high energy | - |
| Dead channel fraction | 0 | [0, 2%] | - |
| Pile-up at high luminosity | 0 | [0, 5%] | - |

---

### 13.5 Radio Aperture Synthesis (`radio_astronomy`)

**Canonical DAG**: F → S → D | **Carrier**: RF | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Antenna gain error | 0 | [0, 5%] per antenna | - |
| Phase offset | 0 | [0, 10] | deg per baseline |
| RFI contamination | 0 | [0, 5%] | of bandwidth |
| uv-coverage gap | none | [0, 20%] | of uv-plane |

---

### 13.6 Acoustic Emission Testing (`acoustic_emission`)

**Canonical DAG**: P → S → D | **Carrier**: Acoustic | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Wave velocity anisotropy | 0 | [0, 15%] | - |
| Sensor coupling variation | 1.0 | [0.5, 1.5] per sensor | - |
| Threshold setting | optimal | +/- 6 | dB |
| Source type misclassification | 0 | [0, 10%] | - |

---

### 13.7 Magnetic Particle Imaging (`magnetic_particle`)

**Canonical DAG**: M → F → D | **Carrier**: Magnetic | **Current maturity**: M0

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| System function error | 0 | [0, 10%] | - |
| Relaxation effect | none | [0, 20%] | peak broadening |
| Background signal | 0 | [0, 5%] | - |
| Drive field amplitude error | 0 | [-5%, 5%] | - |

---

### 13.8 Electrical Impedance Tomography (`impedance_tomo`)

**Canonical DAG**: M → D | **Carrier**: Electric | **Current maturity**: M0

#### B1: Design

**Example prompt**: "Design EIT for lung ventilation monitoring: 16 electrodes on thorax, 10 kHz excitation, adjacent drive pattern, 50 fps."

#### B2: Mismatch parameters

| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Contact impedance | 100 | [50, 500] | ohm per electrode |
| Electrode position error | 0 | [0, 5] | mm |
| Body shape model error | elliptical | +/- 10% radii | - |
| Stray capacitance | 0 | [0, 10] | pF |

**Improvement suggestions**:
- Add absolute imaging benchmark (not just time-difference)
- Include multi-frequency EIT for tissue characterization

---

## 14. Global Improvement Recommendations

### 14.1 Per-Maturity-Level Actions

| Current Level | Action to Reach Next Level | Count of Modalities |
|---------------|---------------------------|--------------------:|
| **M0 → M1** | Define mismatch param ranges; run synthetic single-param test | 78 |
| **M1 → M2** | Add compound mismatch (3+ params); measure rho | 8 |
| **M2 → M3** | Obtain real experimental datasets; validate True-Spec | 4 |
| **M3 → M4** | Add Red Team adversarial injection; live-lab feedback loop | 7 |

### 14.2 Cross-Modality Benchmark Templates

To efficiently bring 78 M0 modalities to M1, use these **category-level templates**:

| Category | B2 Template | B3 Template |
|----------|-------------|-------------|
| **C → D** (deconvolution family) | PSF sigma mismatch + noise level + background | Estimate PSF + noise model |
| **Pi → D** (tomography family) | Center-of-rotation + angular error + noise | Estimate geometry parameters |
| **M → F → S → D** (MRI family) | Coil sensitivity + trajectory + B0 | Estimate coil maps + field map |
| **M → P → D** (ptychographic family) | Probe position + aberration + coherence | Estimate positions + aberrations |
| **P → D** (propagation family) | Speed/propagation error + attenuation + noise | Estimate propagation parameters |
| **M → W → Sigma → D** (spectral family) | Mask/dispersion shift + gain + noise | Estimate mask params + dispersion |

### 14.3 Priority Queue: Next 10 Modalities to Validate

Based on user community size, scientific impact, and benchmark readiness:

| Priority | Modality | Current Level | Why | Key Dataset Needed |
|----------|----------|:---:|-----|-------------------|
| 1 | `ultrasound` | M1 | Largest medical user base after CT/MRI | Plane-wave US benchmark |
| 2 | `pet` | M1 | Nuclear medicine cornerstone | NEMA phantom data |
| 3 | `cryo_em` | M0 | High-impact structural biology | EMPIAR benchmark set |
| 4 | `sim` | M2 | Active super-resolution community | BioSR with pattern metadata |
| 5 | `oct` | M1 | Ophthalmology standard of care | Retinal OCT with calibration |
| 6 | `sar` | M0 | Large remote sensing community | Sentinel-1 SLC data |
| 7 | `fpm` | M1 | Growing computational microscopy | LED array calibration data |
| 8 | `holography` | M1 | Fundamental coherent imaging | Off-axis hologram with ref calibration |
| 9 | `industrial_ct` | M0 | Manufacturing QA critical | GE/Nikon industrial phantom |
| 10 | `hyperspectral_remote` | M0 | Environmental monitoring | AVIRIS/EnMAP benchmark |

---

## 15. Summary Statistics

| Category | Modalities | M0 | M1 | M2 | M3 | M4 |
|----------|:----------:|:--:|:--:|:--:|:--:|:--:|
| Microscopy | 16 | 12 | 3 | 1 | 0 | 0 |
| Compressive Imaging | 4 | 0 | 1 | 0 | 3 | 0 |
| Medical Imaging | 25 | 19 | 4 | 0 | 2 | 0 |
| Coherent Imaging | 3 | 1 | 1 | 0 | 1 | 0 |
| Comp. Photography | 2 | 1 | 0 | 0 | 1 | 0 |
| Comp. Optics | 2 | 2 | 0 | 0 | 0 | 0 |
| Neural Rendering | 2 | 1 | 1 | 0 | 0 | 0 |
| Electron Microscopy | 8 | 8 | 0 | 0 | 0 | 0 |
| Depth Imaging | 3 | 3 | 0 | 0 | 0 | 0 |
| Remote Sensing | 8 | 8 | 0 | 0 | 0 | 0 |
| Industrial Inspection | 8 | 8 | 0 | 0 | 0 | 0 |
| Scientific Instr. | 8 | 8 | 0 | 0 | 0 | 0 |
| Broader Exp. Science | 8 | 8 | 0 | 0 | 0 | 0 |
| **Total** | **97** | **79** | **10** | **1** | **7** | **0** |

**Path to full validation**: 79 modalities at M0 need synthetic benchmarks first. The 7 M3 modalities (CASSI, CACTI, SPC, CT, Ptychography, MRI, Lensless) form the validated core. Target: 20 modalities at M2+ by Phase B, 40+ at M2+ by Phase C.

---

## 16. References

1. PWM Flagship Paper — Typed Primitives and OperatorGraph IR
2. [PWM Targeting System](targeting_system.md) — LIP Arena specification
3. [PWM Imaging Modality Registry](imaging_modalities.md) — 64-modality registry
4. [PWM Modality Standards](modality_standards.md) — Detailed standards for 17 modalities
5. [PWM Operator Mode](operator_mode.md) — Calibration pipeline
6. [PWM Canonical Primitives](plan_canonical_primitives.md) — 10 canonical primitives
7. [PWM Medical Physicist Targets](pwm_medical_physicist_targets.md) — Architecture overview
8. [PWM Modality Mismatch Guide](modality_mismatch_guide.md) — Mismatch evidence for 64 modalities
