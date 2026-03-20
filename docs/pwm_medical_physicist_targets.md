# PWM 4-Benchmark Targeting System: Universal Imaging Modality Coverage

> **Document scope**: Defines four benchmarks for every imaging modality in PWM,
> organized as two flowcharts (B1+B2 and B3+B4), each benchmark independently executable.
> Maps them to the PWM 4-level targeting system and the four medical physicist roles.
>
> **Reference**: [PWM Flagship Paper — Typed Primitives](https://github.com/integritynoble/Physics_World_Model/blob/master/papers/pwm_flagship/main.pdf)

---

## 1. Architecture: Two Flowcharts, Four Independent Benchmarks

The 4 benchmarks are organized into **two flowcharts** that represent the two fundamental directions of imaging science. Each benchmark within a flowchart can be executed **independently**, but when run together they form a complete pipeline.

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    FLOWCHART 1: DESIGN PIPELINE (Spec-First)                 ║
║                                                                               ║
║   ┌──────────────────────┐         ┌──────────────────────────────┐          ║
║   │  B1: DESIGN          │         │  B2: SIMULATE + RECONSTRUCT  │          ║
║   │                      │  Spec   │                              │          ║
║   │  Prompt              │────────>│  Spec                        │          ║
║   │  + Original-Spec     │         │  ──> measurement y           │          ║
║   │  ──> Spec            │         │  ──> reconstruction x_hat    │          ║
║   │                      │         │  ──> feedback (improve sys)  │          ║
║   └──────────────────────┘         └──────────────────────────────┘          ║
║   Run independently: YES           Run independently: YES                    ║
║   (B2 can take any Spec,           (B1 can output Spec without               ║
║    not only from B1)                running B2)                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║                  FLOWCHART 2: DATA PIPELINE (Dataset-First)                  ║
║                                                                               ║
║   ┌──────────────────────┐         ┌──────────────────────────────┐          ║
║   │  B3: IDENTIFY        │         │  B4: CORRECT + RECONSTRUCT   │          ║
║   │                      │  Spec   │                              │          ║
║   │  Dataset + Prompt    │────────>│  Dataset + Spec              │          ║
║   │  + Original-Spec     │         │  ──> corrected Spec          │          ║
║   │  ──> Spec            │         │  ──> reconstruction x_hat    │          ║
║   │                      │         │  ──> feedback (improve sys)  │          ║
║   └──────────────────────┘         └──────────────────────────────┘          ║
║   Run independently: YES           Run independently: YES                    ║
║   (B4 can take any Spec,           (B3 can output Spec without               ║
║    not only from B3)                running B4)                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Cross-flowchart connections:
  B1 Spec ──> B4 (use designed Spec as starting point for correction)
  B3 Spec ──> B2 (simulate from an identified Spec to validate it)
  B4 feedback ──> B1 (use correction insights to redesign the system)
  B2 feedback ──> B3 (simulation reveals which parameters matter for identification)
```

### Why Two Flowcharts?

| | Flowchart 1 (B1 + B2) | Flowchart 2 (B3 + B4) |
|---|---|---|
| **Starting point** | Requirements (prompt) | Data (measurement) |
| **Direction** | Forward: design --> simulate | Inverse: data --> identify --> correct |
| **Spec role** | Spec is *generated* | Spec is *inferred* or *corrected* |
| **True-Spec** | Not needed for B1; known for B2 (injected mismatch) | Hidden ground truth for both B3 and B4 |
| **Primary use case** | New system design, protocol optimization | Existing system calibration, QA, troubleshooting |
| **Medical physicist analogy** | Commissioning a new scanner | Annual QA / daily calibration |

### Why Each Benchmark is Independent

Each benchmark has **self-contained inputs and outputs** and can run without the other benchmark in its flowchart:

| Benchmark | Can run without... | How |
|-----------|-------------------|-----|
| **B1** | B2 | Output Spec is evaluated for physical validity and Pareto optimality alone |
| **B2** | B1 | Accept any Spec (user-provided, template, or from B3) as input |
| **B3** | B4 | Output Spec is evaluated against True-Spec for parameter accuracy alone |
| **B4** | B3 | Accept any Spec (user-provided, nominal, or from B1) as input |

---

## 2. The Spec: Built from 11 Typed Primitives

A **Spec** is the machine-readable description of how a dataset is obtained. It is the universal currency across all four benchmarks. A **True-Spec** is the hidden ground-truth Spec used to judge quality in B3 and B4.

Every Spec is a DAG (directed acyclic graph) composed from these **typed primitives**:

| # | Primitive | Symbol | Physics Stage | Role in Spec |
|---|-----------|--------|---------------|--------------|
| 1 | Propagate | P | Propagation | Free-space wave propagation (Fresnel, angular spectrum, acoustic) |
| 2 | Modulate | M | Interaction | Element-wise multiplication (mask, coil sensitivity, absorption) |
| 3 | Project | Pi | Encoding-Projection | Radon line-integral projection (CT, emission tomography) |
| 4 | Encode | F | Encoding-Projection | Fourier-domain encoding (MRI k-space, reciprocal space) |
| 5 | Convolve | C | Propagation / Detection | Spatial convolution (PSF, blur kernel, diffraction) |
| 6 | Accumulate | Sigma | Detection-Readout | Summation over spectral/temporal axis |
| 7 | Detect | D | Detection-Readout | Detector response (linear-intensity, logarithmic, sigmoid, Poisson-rate, coherent-field) |
| 8 | Sample | S | Detection-Readout | Sub-sampling on index set (k-space undersampling, scan positions) |
| 9 | Disperse | W | Detection-Readout | Wavelength-dependent spatial shift (prism, grating) |
| 10 | Scatter | R | Interaction | Direction change and/or energy shift (Compton, Raman, fluorescence) |
| 11 | Source | Src | Generation | Carrier generation (photon source, X-ray tube, RF excitation, electron gun) |

A Spec also declares: noise model, mismatch parameter ranges, scene/sample description, and reconstruction solver configuration. The True-Spec adds the exact values of every mismatch parameter (mask rotation, shift, noise level, gain drift, etc.).

---

## 3. Benchmark Definitions

### Benchmark 1 (B1): Prompt + Original-Spec --> Spec  (DESIGN)

**Flowchart**: 1 (Design Pipeline, first stage)

| Field | Description |
|-------|-------------|
| **Input** | Natural-language prompt describing imaging requirements (e.g., "hyperspectral imaging of tissue at 450-650 nm, 28 bands, snapshot acquisition") + optional Original-Spec from a previous round |
| **Output** | A complete Spec: DAG of typed primitives, system elements (lens, mask, detector), acquisition parameters, noise model, mismatch tolerance envelope |
| **Evaluation** | (a) Physical validity -- does the Spec define a realizable imaging system? (b) Constraint satisfaction -- does it meet the stated requirements? (c) Robustness margin -- how much mismatch can it tolerate before degradation? (d) Calibration cost -- how many GPU-hours to correct from tolerance-edge mismatch? |
| **True-Spec** | Not applicable (generative benchmark); evaluated by forward-simulation feasibility and Pareto optimality |
| **PWM Track** | **Track 4: Design** |
| **Independent?** | YES -- produces a Spec without needing B2. The Spec can be handed to B2, B4, or used directly. |

**What it tests**: Can you design an imaging system from a text description, grounding the design in the typed-primitive vocabulary?

---

### Benchmark 2 (B2): Spec --> Reconstructions + Feedback  (FORWARD + RECONSTRUCT)

**Flowchart**: 1 (Design Pipeline, second stage)

| Field | Description |
|-------|-------------|
| **Input** | A complete Spec (from B1, from a user, or from B3 -- any source) |
| **Output** | (a) Simulated measurement `y` with realistic noise and mismatch injected per the Spec's tolerance envelope, (b) Reconstruction `x_hat` using declared solver, (c) Feedback report: which Triad gate is binding, what mismatch dominates, how to improve the system |
| **Evaluation** | (a) Forward-model fidelity (epsilon_tier2 < 0.01), (b) Reconstruction quality under nominal conditions (Scenario I), (c) Reconstruction quality under mismatch (Scenario II), (d) Feedback quality -- are suggestions actionable? |
| **True-Spec** | The Spec itself is ground truth for forward simulation; the injected mismatch parameters are known. The Spec contains all the things needed to build the imaging system -- the Spec can directly produce the imaging system. |
| **PWM Track** | **Track 1: Correct** (Scenario I-II baseline) + **Track 2: Diagnose** (feedback) |
| **Independent?** | YES -- accepts any Spec as input, not only output of B1. |

**What it tests**: Given a complete system description, can you faithfully simulate it, reconstruct, and diagnose what limits performance? The feedback should consider all noise and mismatches and recommend how to improve the system.

---

### Benchmark 3 (B3): Dataset + Prompt + Original-Spec --> Spec  (SYSTEM IDENTIFICATION)

**Flowchart**: 2 (Data Pipeline, first stage)

| Field | Description |
|-------|-------------|
| **Input** | Real measurement dataset (measurement `y`, system matrix, metadata) + natural-language prompt + optional Original-Spec from a previous round |
| **Output** | An inferred Spec that explains how the dataset was obtained -- the DAG of primitives, estimated system parameters, noise model, mismatch characterization |
| **Evaluation** | Compare inferred Spec against hidden True-Spec: (a) Parameter RMSE for all mismatch parameters, (b) Correct identification of the primitive DAG, (c) Correct noise model identification, (d) Uncertainty calibration (do declared CIs contain truth?) |
| **True-Spec** | Hidden ground truth containing all exact parameters: mask rotation degree, shift offsets, noise level, gain drift, dispersion coefficients, etc. The True-Spec contains all the parameters of how to get the dataset, including mask rotation degree and shift and noise level, which can recover the real mismatch and correct it easily. Contestants know the *range* of mismatch but not the exact values. |
| **PWM Track** | **Track 3: No-GT** + **Track 2: Diagnose** |
| **Independent?** | YES -- produces a Spec without needing B4. The Spec can then be handed to B4 or B2. |

**What it tests**: Given real data and a rough description, can you reverse-engineer the imaging system -- identifying what primitives were used and what their exact parameters are?

---

### Benchmark 4 (B4): Dataset + Spec --> Correction + Reconstruction + Feedback  (CORRECT + DIAGNOSE)

**Flowchart**: 2 (Data Pipeline, second stage)

| Field | Description |
|-------|-------------|
| **Input** | Real measurement dataset `y` + an imperfect Spec (from B3, from a user, or from B1 -- any source). The Spec may have errors within the declared tolerance envelope. |
| **Output** | (a) Corrected Spec (calibrated operator parameters), (b) Reconstruction `x_hat` using corrected operator, (c) Feedback: TriadReport (dominant gate, evidence, confidence), suggestions for improving the Spec and the real experimental system |
| **Evaluation** | Compare against hidden True-Spec: (a) Recovery ratio rho, (b) Parameter recovery RMSE, (c) TriadReport accuracy, (d) Feedback quality -- would suggestions improve the real system? (e) Compute efficiency RoIC (dB/GPU-hr). The True-Spec should also consider how to improve the spec and the real experimental system. |
| **True-Spec** | Hidden ground truth containing exact parameters + recommendations for system improvement. Used to judge both correction accuracy and feedback quality. |
| **PWM Track** | **Track 1: Correct** (primary) + **Track 2: Diagnose** (feedback) + **Track 3: No-GT** (when no GT reconstruction available) |
| **Independent?** | YES -- accepts any Spec as input, not only output of B3. |

**What it tests**: The full ISA pipeline -- can you fix what's wrong with the system model, reconstruct well, and provide actionable guidance for improving the physical instrument?

---

## 4. Mapping to PWM 4-Level Targeting System

### 4.1 Mapping to Evaluation Tracks

| Benchmark | Track 1: Correct | Track 2: Diagnose | Track 3: No-GT | Track 4: Design |
|-----------|:-:|:-:|:-:|:-:|
| **B1** (Prompt-->Spec) | | | | **PRIMARY** |
| **B2** (Spec-->Recon) | Scenario I-II baseline | Feedback/attribution | | Design validation |
| **B3** (Data-->Spec) | | Parameter identification | **PRIMARY** | |
| **B4** (Data+Spec-->Correction) | **PRIMARY** | Feedback | When no GT available | |

### 4.2 Mapping to Maturation Phases

| Phase | Timeline | Benchmarks Active | Key Milestone |
|-------|----------|-------------------|---------------|
| **Phase A** (Internal) | 0-6 months | B2, B4 on 7 validated modalities | Sealed-simulator B2+B4 for CASSI, CACTI, SPC, CT, Ptychography, MRI, Lensless |
| **Phase B** (Pilot) | 6-12 months | B1-B4 on 10+ modalities | B1 and B3 added; first live-lab True-Specs; external submissions |
| **Phase C** (Full) | 12-24 months | B1-B4 on 20+ modalities | All benchmarks operational with Red Team adversarial injection |
| **Phase D** (Utility) | 24+ months | B1-B4 on 168+ modalities | Hardware-in-the-loop; B1 from natural language to validated pipeline in minutes |

### 4.3 Mapping to Maturation Levels (L0-L5)

| Level | B1 Capability | B2 Capability | B3 Capability | B4 Capability |
|-------|---------------|---------------|---------------|---------------|
| **L0** (Muddle) | No spec language | Ad-hoc simulation | Manual system ID | Manual calibration |
| **L1** (Measurable) | Spec templates exist | Metrics defined, comparable | Parameter ranges known | Recovery ratio tracked |
| **L2** (Repeatable) | SOP for spec writing | Documented forward pipelines | Documented ID procedures | Calibration SOPs codified |
| **L3** (Automated) | Auto-spec from prompt | Auto-simulation + diagnosis | Auto-identification from data | Auto-correction, rho >= 0.80 |
| **L4** (Industrialized) | Design-as-a-service | Simulation-as-a-service | Identification-as-a-service | Calibration-as-a-service |
| **L5** (Commodity) | Any system self-designs | Any system self-simulates | Any system self-identifies | Any system self-corrects |

---

## 5. Complete Modality Registry with 4-Benchmark Specifications

For every modality: **B1** = Design focus, **B2** = Forward/Recon focus, **B3** = Identification focus, **B4** = Correction focus.

### 5.1 Microscopy (24 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 1 | `widefield` | Widefield Fluorescence | C --> D | Photon | PSF design, NA selection, illumination uniformity | Deconvolution under PSF mismatch and Poisson noise | Estimate PSF sigma, background level, gain | Correct PSF model, suppress out-of-focus blur |
| 2 | `widefield_lowdose` | Low-Dose Widefield | C --> D | Photon | Photon budget vs SNR tradeoff | Denoise under extreme Poisson noise | Estimate photon rate, read noise, background | Correct noise model, denoise-then-deconvolve |
| 3 | `confocal_livecell` | Confocal Live-Cell | C --> D | Photon | Pinhole size, scan speed vs photobleaching | Deconvolution + motion correction | Estimate drift rate, PSF, bleaching curve | Correct drift, update PSF for live conditions |
| 4 | `confocal_3d` | Confocal 3D Z-Stack | C --> D | Photon | Axial vs lateral resolution, z-step, depth attenuation | 3D deconvolution under depth-dependent PSF | Estimate 3D PSF, refractive index, attenuation | Correct depth-dependent aberrations |
| 5 | `sim` | Structured Illumination (SIM) | M --> C --> D | Photon | Pattern frequency, orientations, phase steps | Wiener-SIM reconstruction under pattern mismatch | Estimate pattern freq, phase, modulation depth | Correct illumination pattern errors |
| 6 | `lightsheet` | Light-Sheet (LSFM) | C --> D | Photon | Sheet thickness, detection NA, multi-view config | Destripe + deconvolution + multi-view fusion | Estimate stripe strength, sheet profile, tilt | Correct sheet alignment, remove stripe artifacts |
| 7 | `flim` | Fluorescence Lifetime (FLIM) | M --> R --> D | Photon | Excitation pulse, time gates, lifetime range | Phasor analysis under IRF mismatch | Estimate IRF width, background, afterpulsing | Correct IRF, calibrate lifetime axis |
| 8 | `fpm` | Fourier Ptychographic (FPM) | M --> P --> D | Photon | LED array geometry, overlap ratio, NA synthesis | Phase retrieval under LED position error | Estimate LED positions, aberrations, intensity | Correct LED misalignment, pupil aberration |
| 9 | `two_photon` | Two-Photon / Multiphoton | C --> D | Photon | Excitation wavelength, NA, scan pattern, depth | Deconvolution under depth-dependent scattering | Estimate scattering coefficient, PSF vs depth | Correct depth-dependent PSF and attenuation |
| 10 | `sted` | STED Microscopy | C --> D | Photon | Depletion beam shape, saturation power, resolution | Deconvolution with effective PSF under STED mismatch | Estimate depletion efficiency, effective resolution | Correct depletion beam alignment |
| 11 | `palm_storm` | PALM/STORM Localization | M --> D | Photon | Label density, photon budget, frame count | Localization under drift and background mismatch | Estimate drift trajectory, photon rate, background | Correct drift, re-localize with updated model |
| 12 | `tirf` | TIRF Microscopy | C --> D | Photon | Incidence angle, evanescent depth, NA | Deconvolution with evanescent-field PSF | Estimate penetration depth, angle, background | Correct angle calibration, background subtraction |
| 13 | `polarization` | Polarization Microscopy | M --> C --> D | Photon | Analyzer angles, retardance range, Stokes config | Mueller matrix reconstruction under calibration error | Estimate retardance offset, polarizer extinction | Correct polarization calibration |
| 14 | `expansion` | Expansion Microscopy | C --> D | Photon | Expansion factor, gel uniformity, labeling | Deconvolution correcting non-uniform expansion | Estimate local expansion factor, distortion field | Correct expansion non-uniformity |
| 15 | `minflux` | MINFLUX Nanoscopy | C --> D | Photon | Beam pattern, localization precision, photon budget | Localization under beam misalignment | Estimate beam center position, photon rate | Correct beam positioning errors |
| 16 | `ism` | Image Scanning Microscopy | C --> D | Photon | Detector array geometry, reassignment strategy | Pixel reassignment under geometric distortion | Estimate detector offset, magnification error | Correct pixel reassignment parameters |
| 17 | `phase_contrast` | Phase Contrast Microscopy | C --> D | Photon | Phase ring design, condenser annulus, NA match | Halo artifact correction under ring misalignment | Estimate phase ring position, absorption ratio | Correct phase ring alignment, halo suppression |
| 18 | `dic` | Differential Interference Contrast | M --> C --> D | Photon | Prism shear, bias retardation, NA selection | DIC gradient reconstruction under bias error | Estimate shear distance, bias offset, extinction ratio | Correct bias retardation, prism alignment |
| 19 | `dark_field` | Dark-Field Microscopy | C --> D | Photon | Stop size, illumination angle, NA partitioning | Dark-field recon under illumination non-uniformity | Estimate stop alignment, illumination asymmetry | Correct stop positioning, flat-field correction |
| 20 | `lattice_lightsheet` | Lattice Light-Sheet | C --> D | Photon | Lattice pattern, dithering, z-step, tile overlap | Deconvolution + tile stitching under lattice error | Estimate lattice period, dither range, tile offset | Correct lattice parameters, stitch registration |
| 21 | `shg` | Second Harmonic Generation (SHG) | M --> R --> D | Photon | Excitation wavelength, polarization, NA, detection filter | SHG reconstruction under phase-matching error | Estimate SHG efficiency, polarization orientation | Correct phase-matching, polarization calibration |
| 22 | `spinning_disk` | Spinning Disk Confocal | C --> D | Photon | Pinhole spacing, rotation speed, camera sync | Deconvolution under pinhole crosstalk | Estimate pinhole spacing error, crosstalk, sync lag | Correct crosstalk, synchronization timing |
| 23 | `three_photon` | Three-Photon Microscopy | C --> D | Photon | Excitation 1300/1700 nm, repetition rate, depth | Deconvolution under deep-tissue scattering | Estimate scattering length, pulse broadening | Correct depth-dependent attenuation and PSF |
| 24 | `dna_paint` | DNA-PAINT Super-Resolution | M --> D | Photon | Imager strand concentration, binding kinetics | Localization under binding rate variation | Estimate on-rate, off-rate, background binding | Correct binding kinetics model, background |

---

### 5.2 Compressive Imaging (4 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 25 | `cassi` | CASSI (Spectral) | M --> W --> Sigma --> D | Photon | Mask pattern, dispersion element, spectral range, spatial resolution | GAP-TV / MST-L under mask shift (dx,dy), rotation (theta), dispersion slope (a1), axis offset (alpha) | Estimate 5 mismatch params from measurement residuals | Correct all 5 params; rho validated at 85% (flagship) |
| 26 | `spc` | Single-Pixel Camera | M --> Sigma --> D | Photon | Pattern type (Hadamard/Gaussian), sampling rate, DMD resolution | FISTA-TV under gain drift (alpha) and measurement noise (sigma_y) | Estimate gain drift curve and noise level | Correct gain model; rho validated at 86% |
| 27 | `cacti` | CACTI (Temporal) | M --> Sigma --> D | Photon | Mask shift type, compression ratio, temporal resolution | GAP-TV under spatial shift, rotation, temporal clock, gain, offset | Estimate 8 mismatch params from temporal correlations | Correct mask replication errors; rho validated at 100% |
| 28 | `matrix` | Generic Matrix Sensing | M --> D | Photon | Measurement matrix design (RIP, coherence), conditioning | CG/ADMM under matrix perturbation | Estimate matrix condition number, perturbation magnitude | Correct matrix calibration errors |

**Validated Baselines** (flagship paper): CASSI GAP-TV +0.76 dB rho=85%; CACTI GAP-TV +10.21 dB rho=100%; SPC FISTA-TV +7.71 dB rho=86%; CT FBP +10.68 dB rho=100%; Ptychography ePIE +7.09 dB rho=100%; MRI SENSE +1.75-7.14 dB; Lensless ADMM +3.55 dB rho=78%.

---

### 5.3 Medical Imaging (37 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 29 | `ct` | X-ray CT | Pi --> D | X-ray | Geometry (fan/parallel/cone), angles, detector count, dose | FBP/SART under center-of-rotation offset, angular offset, detector tilt, beam hardening | Estimate CoR offset, angular errors, hardening coefficients | Correct geometry; rho=100%, +10.68 dB |
| 30 | `mri` | MRI | M --> F --> S --> D | Spin/RF | Coil count, trajectory (Cartesian/radial/spiral), acceleration factor | SENSE/GRAPPA under coil sensitivity error, k-space trajectory deviation, off-resonance | Estimate coil maps, trajectory errors, field map | Correct coil + trajectory; +1.75-7.14 dB |
| 31 | `xray_radiography` | X-ray Radiography | Pi --> D | X-ray | Source-detector distance, filtration, exposure parameters | TV-FISTA under scatter, beam hardening, detector lag | Estimate scatter fraction, hardening polynomial | Correct scatter, beam hardening correction |
| 32 | `ultrasound` | Ultrasound B-mode | P --> D | Acoustic | Transducer array, frequency, focus depth, apodization | DAS beamforming under speed-of-sound error, phase aberration | Estimate sound speed profile, aberration screen | Correct aberration, adaptive beamforming |
| 33 | `pet` | PET | Pi --> D | Gamma | Crystal ring geometry, TOF resolution, attenuation correction | MLEM/OSEM under attenuation map error, scatter, randoms, normalization | Estimate attenuation factors, scatter fraction, normalization table | Correct attenuation, scatter, normalization |
| 34 | `spect` | SPECT | Pi --> D | Gamma | Collimator type, orbit, energy window, attenuation correction | MLEM with depth-dependent resolution under collimator response error | Estimate collimator params, attenuation map, center-of-rotation | Correct CoR, collimator model, attenuation |
| 35 | `fluoroscopy` | Fluoroscopy | Pi --> D | X-ray | Frame rate, dose per frame, detector type | TV-FISTA under temporal lag, scatter, geometric distortion | Estimate lag coefficient, scatter model, pincushion distortion | Correct lag, flat-field, geometric distortion |
| 36 | `mammography` | Mammography | Pi --> D | X-ray | Target/filter combination, compression, detector type | TV-FISTA under scatter, heel effect, detector MTF variation | Estimate scatter-to-primary ratio, heel effect profile | Correct scatter, MTF correction |
| 37 | `dexa` | DEXA | Pi --> D | X-ray | Dual energy selection, scan mode, calibration phantom | Dual-energy decomposition under beam hardening, fat-lean mismatch | Estimate effective energies, calibration polynomial | Correct calibration, decomposition coefficients |
| 38 | `cbct` | Cone-Beam CT (CBCT) | Pi --> D | X-ray | Cone angle, flat-panel geometry, rotation arc, dose | FDK under cone-beam artifacts, scatter, truncation | Estimate scatter fraction, truncation extent, detector offset | Correct scatter, extend FOV, ring artifacts |
| 39 | `angiography` | X-ray Angiography | Pi --> D | X-ray | Contrast timing, frame rate, subtraction protocol | DSA subtraction under patient motion, misregistration | Estimate motion field between mask and contrast frames | Correct motion-compensated subtraction |
| 40 | `dot` | Diffuse Optical Tomography | M --> R,P,R --> D | Photon | Source-detector layout, wavelength selection, time/frequency domain | Born approximation inversion under scattering coefficient error | Estimate absorption/scattering coefficients, boundary conditions | Correct optical properties, boundary model |
| 41 | `photoacoustic` | Photoacoustic | M --> P --> D | Acoustic | Transducer array, laser wavelength, fluence model | Backprojection under speed-of-sound heterogeneity, acoustic attenuation | Estimate sound speed map, Grueneisen parameter, fluence | Correct sound speed model, fluence compensation |
| 42 | `oct` | OCT | P+P --> Sigma --> D | Photon | Source bandwidth, reference arm, scan pattern, axial resolution | FFT recon under dispersion mismatch, reference arm drift | Estimate dispersion coefficients, reference arm position | Correct dispersion, reference drift |
| 43 | `fmri` | Functional MRI (BOLD) | M --> F --> S --> D | Spin/RF | TR/TE, spatial resolution, temporal resolution, EPI trajectory | SENSE + GLM under geometric distortion, signal dropout | Estimate field map, distortion, motion parameters | Correct distortion, motion, physiological noise |
| 44 | `mrs` | MR Spectroscopy | M --> F --> S --> D | Spin/RF | Voxel localization, spectral bandwidth, water suppression | LCModel fitting under lineshape distortion, baseline error | Estimate lineshape, eddy current phase, residual water | Correct lineshape, eddy current, baseline |
| 45 | `diffusion_mri` | Diffusion MRI (DTI) | M --> F --> S --> D | Spin/RF | b-values, gradient directions, eddy currents | WLS tensor fitting under gradient nonlinearity, eddy current distortion | Estimate gradient tables, eddy current coefficients | Correct gradient nonlinearity, eddy currents |
| 46 | `doppler_ultrasound` | Doppler Ultrasound | P --> D | Acoustic | PRF, wall filter, velocity range, angle of insonation | Autocorrelation estimator under aliasing, wall filter error | Estimate flow angle, PRF aliasing threshold, clutter | Correct angle, anti-aliasing, clutter filter |
| 47 | `elastography` | Shear-Wave Elastography | P --> D | Acoustic | Push pulse, tracking method, shear wave frequency | TOF inversion under wave reflection, dispersion | Estimate shear wave speed, attenuation, boundary effects | Correct reflection, dispersion compensation |
| 48 | `endoscopy` | Fiber Bundle Endoscopy | M --> C --> D | Photon | Fiber count, FOV, bending radius, illumination | TV-FISTA under fiber cross-talk, non-uniform transmission | Estimate fiber transmission map, geometric distortion | Correct fiber calibration, distortion |
| 49 | `fundus` | Fundus Camera | C --> D | Photon | FOV, illumination wavelength, mydriasis | Richardson-Lucy under aberration, non-uniform illumination | Estimate aberration coefficients, illumination profile | Correct aberrations, flat-field |
| 50 | `octa` | OCT Angiography | P+P --> Sigma --> D | Photon | Scan density, interscan time, decorrelation method | TV-FISTA under bulk motion, projection artifact | Estimate bulk motion, shadow artifacts | Correct bulk motion, projection artifacts |
| 51 | `proton_therapy_img` | Proton Therapy Imaging | Pi --> D | Proton | Energy range, detector stack, range verification | Backprojection under range uncertainty, scattering | Estimate water-equivalent path length, scattering model | Correct range model, scattering compensation |
| 52 | `brachytherapy_img` | Brachytherapy Imaging | Pi --> D | Gamma/X-ray | Source geometry, applicator model, imaging protocol | TG-43/TG-186 dose with imaging verification | Estimate source position, applicator geometry | Correct source localization, applicator model |
| 53 | `portal_imaging` | Portal Imaging (EPID) | Pi --> D | MV X-ray | Detector geometry, gantry angle, field size | Backprojection under sag, flex, MLC position error | Estimate gantry sag, detector offset, MLC positions | Correct geometric calibration, MLC model |
| 54 | `spectral_ct` | Photon-Counting Spectral CT | Pi --> W --> D | X-ray | Energy bins, threshold settings, pile-up correction | Material decomposition under threshold drift, charge sharing | Estimate energy thresholds, charge sharing width | Correct threshold calibration, pile-up |
| 55 | `mr_elastography` | MR Elastography | M --> F --> S --> D | Spin/RF | Driver frequency, MEG encoding, inversion algorithm | LFE/DI under wave attenuation, reflection, boundary | Estimate shear modulus, attenuation, wave speed | Correct wave model, boundary conditions |
| 56 | `cest_mri` | CEST MRI | M --> F --> S --> D | Spin/RF | Saturation frequency, power, B0 mapping | Z-spectrum fitting under B0/B1 inhomogeneity | Estimate B0 map, B1 map, CEST asymmetry | Correct B0/B1 correction, water reference |
| 57 | `asl_mri` | Arterial Spin Labeling | M --> F --> S --> D | Spin/RF | Labeling scheme, PLD, background suppression | Perfusion quantification under transit time error | Estimate bolus arrival time, labeling efficiency | Correct transit time, partial volume |
| 58 | `mra` | MR Angiography | M --> F --> S --> D | Spin/RF | TOF vs CE vs PC, spatial resolution, coverage | MIP/SSD under flow artifact, vessel signal | Estimate flow velocity, vessel boundary | Correct flow artifacts, background suppression |
| 59 | `swi` | Susceptibility-Weighted Imaging | M --> F --> S --> D | Spin/RF | TE, filter size, phase mask | SWI reconstruction under field inhomogeneity | Estimate susceptibility sources, field map | Correct background field, phase unwrapping |
| 60 | `mr_fingerprinting` | MR Fingerprinting | M --> F --> S --> D | Spin/RF | Flip angle pattern, TR pattern, dictionary design | Dictionary matching under B0/B1 variation, aliasing | Estimate T1, T2, B0, B1 maps simultaneously | Correct B0/B1, refine dictionary matching |
| 61 | `ivus` | Intravascular Ultrasound | P --> D | Acoustic | Transducer frequency (20-60 MHz), pullback speed | Polar reconstruction under NURD, catheter eccentricity | Estimate NURD profile, catheter position offset | Correct NURD, re-center catheter model |
| 62 | `ceus` | Contrast-Enhanced Ultrasound | P --> R --> D | Acoustic | MI, pulse scheme, frame rate, contrast agent | Contrast-specific imaging under tissue clutter | Estimate tissue signal, bubble nonlinearity curve | Correct tissue subtraction, linearize contrast |
| 63 | `digital_breast_tomo` | Digital Breast Tomosynthesis | Pi --> D | X-ray | Angular range, projection count, detector type | Backprojection under geometric calibration error | Estimate tube positions, detector flex, compression | Correct geometry, reduce out-of-plane artifacts |
| 64 | `confocal_endomicroscopy` | Confocal Endomicroscopy | M --> C --> D | Photon | Fiber bundle, laser wavelength, frame rate | Deconvolution under fiber honeycomb pattern | Estimate fiber core positions, coupling efficiency | Correct fiber pattern, interpolation artifacts |
| 65 | `nirs_brain` | Functional NIRS | M --> R,P --> D | Photon | Source-detector distance, wavelengths, sampling rate | Modified Beer-Lambert under scalp coupling variation | Estimate DPF, coupling coefficients, motion artifact | Correct motion, coupling, superficial signal |

---

### 5.4 Coherent Imaging (5 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 66 | `ptychography` | Ptychographic Imaging | M --> P --> D | Electron/Photon | Probe size, overlap ratio, scan pattern, coherence | ePIE under probe position error, defocus, aberration | Estimate probe positions, aberration coefficients | Correct positions; rho=100%, +7.09 dB |
| 67 | `holography` | Digital Holographic Microscopy | P --> D | Photon | Reference beam angle, wavelength, off-axis vs inline | Angular spectrum under reference beam angle error, vibration | Estimate carrier frequency, reference angle, phase offset | Correct reference beam model, vibration |
| 68 | `phase_retrieval` | Coherent Diffractive Imaging | P --> D | Photon/Electron | Support constraint, oversampling ratio, coherence | HIO/ER under support error, partial coherence | Estimate support boundary, coherence function | Correct support, coherence model |
| 69 | `odt` | Optical Diffraction Tomography | P --> D | Photon | Illumination angles, refractive index range, NA | Rytov/Born inversion under missing cone, RI drift | Estimate RI map, illumination angle errors | Correct missing cone, RI calibration |
| 70 | `talbot_lau` | Talbot-Lau X-ray Interferometry | M --> P --> D | X-ray | Grating periods, design energy, inter-grating distance | Phase stepping under grating misalignment, vibration | Estimate grating positions, period mismatch, visibility | Correct grating alignment, period matching |

---

### 5.5 Computational Photography (5 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 71 | `lensless` | Lensless / Diffuser Camera | C --> D | Photon | Diffuser/mask type, sensor distance, PSF calibration | ADMM under PSF shift, scale drift, defocus | Estimate PSF shift, scale, defocus offset | Correct PSF model; rho=78%, +3.55 dB |
| 72 | `panorama` | Panorama Multi-Focus Fusion | C --> D | Photon | Focal sweep range, focal planes, depth of field | Laplacian pyramid under focus distance error | Estimate focal distances, aperture, depth map | Correct focal plane registration |
| 73 | `coded_exposure` | Coded Exposure / Flutter Shutter | M --> C --> D | Photon | Shutter code, exposure time, motion range | Deblurring under code timing error, unknown motion | Estimate shutter function, motion kernel | Correct shutter timing, motion model |
| 74 | `event_camera` | Event Camera / DVS | M --> D | Photon | Contrast threshold, temporal resolution, bias settings | Event-to-frame under threshold mismatch, refractory period | Estimate threshold per pixel, refractory time | Correct threshold calibration, hot pixels |
| 75 | `hdr_imaging` | HDR Imaging | M --> Sigma --> D | Photon | Exposure bracketing, tone mapping, dynamic range | HDR merge under exposure time error, ghosting | Estimate exposure ratios, CRF, motion between frames | Correct exposure calibration, deghosting |

---

### 5.6 Computational Optics (2 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 76 | `light_field` | Light Field Imaging | C --> S --> D | Photon | Microlens array, angular vs spatial resolution | Shift-and-sum under microlens alignment error | Estimate microlens pitch, rotation, f-number | Correct microlens calibration |
| 77 | `integral` | Integral Photography | C --> S --> D | Photon | Lens array geometry, baseline, depth range | Depth estimation under lens distortion | Estimate lens positions, distortion coefficients | Correct geometric calibration |

---

### 5.7 Neural Rendering (2 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 78 | `nerf` | Neural Radiance Fields | M --> P --> D | Photon | View count, camera placement, scene bounds | NeRF/Instant-NGP under camera pose error, intrinsic error | Estimate camera poses, focal length, distortion | Correct camera calibration, refine poses |
| 79 | `gaussian_splatting` | 3D Gaussian Splatting | M --> P --> D | Photon | Initial point cloud, densification, view selection | 3DGS under SfM initialization error | Estimate point cloud quality, initialization bias | Correct initialization, re-densify |

---

### 5.8 Electron Microscopy (11 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 80 | `sem` | SEM | C --> D | Electron | Beam energy, working distance, detector type | Direct imaging under charging, drift, astigmatism | Estimate stigmation, working distance, drift rate | Correct astigmatism, drift compensation |
| 81 | `tem` | TEM | C --> D | Electron | Acceleration voltage, aperture, defocus series | CTF correction under defocus, astigmatism, beam tilt | Estimate CTF params (defocus, Cs, astigmatism) | Correct CTF, aberration model |
| 82 | `electron_tomography` | Electron Tomography | Pi --> D | Electron | Tilt range, tilt increment, missing wedge | SIRT/WBP under tilt axis misalignment, magnification change | Estimate tilt axis offset, magnification variation | Correct tilt axis, missing wedge |
| 83 | `stem` | STEM | S --> D | Electron | Convergence angle, detector geometry, scan pattern | Direct imaging under scan distortion, probe aberration | Estimate scan distortion, probe parameters | Correct scan calibration |
| 84 | `electron_diffraction` | 4D-STEM Diffraction | M --> P --> D | Electron | Probe size, scan step, camera length | Ptychographic recon under camera length error | Estimate camera length, beam center, rotation | Correct geometry calibration |
| 85 | `ebsd` | EBSD | R --> D | Electron | Tilt angle, step size, detector geometry | Hough indexing under pattern center error | Estimate pattern center (PC), detector tilt | Correct PC calibration |
| 86 | `eels` | EELS | S --> D | Electron | Energy range, dispersion, collection angle | Fourier ratio under energy drift, gain variation | Estimate energy drift, gain instability | Correct energy calibration, gain |
| 87 | `electron_holography` | Electron Holography | P --> D | Electron | Biprism voltage, fringe spacing, FOV | Fourier sideband under biprism drift | Estimate biprism voltage drift, fringe rotation | Correct fringe analysis parameters |
| 88 | `cryo_et` | Cryo-Electron Tomography | Pi --> D | Electron | Tilt scheme, dose fractionation, defocus | SIRT under beam-induced motion, dose damage | Estimate per-tilt motion, CTF, accumulated dose | Correct motion, CTF per tilt, dose weighting |
| 89 | `fib_sem` | FIB-SEM | S --> C --> D | Electron + Ion | Milling current, slice thickness, imaging kV | 3D stack under curtaining, charging, slice thickness variation | Estimate slice thickness, curtain artifact, charging | Correct curtaining, align slices, normalize intensity |
| 90 | `edx_mapping` | STEM-EDX Elemental Mapping | M --> R --> D | Electron | Beam current, dwell time, detector solid angle | Element quantification under absorption, fluorescence | Estimate absorption correction, fluorescence yield | Correct absorption, cliff-lorimer factors |

---

### 5.9 Depth Imaging (5 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 91 | `tof_camera` | Time-of-Flight Camera | P --> D | Photon/IR | Modulation frequency, integration time, multi-path | TV-FISTA under multi-path interference, phase wrap | Estimate multi-path coefficients, wrap count | Correct multi-path, phase unwrapping |
| 92 | `lidar` | LiDAR Scanner | P --> S --> D | Photon | Scan pattern, pulse rate, wavelength, range | Point cloud recon under timing jitter, angular error | Estimate timing calibration, angular encoder error | Correct timing, angular calibration |
| 93 | `structured_light` | Structured-Light 3D | M --> C --> D | Photon | Pattern type, projector-camera geometry | Phase unwrapping under defocus, gamma nonlinearity | Estimate gamma curve, projector-camera extrinsics | Correct gamma, geometric calibration |
| 94 | `photometric_stereo` | Photometric Stereo | M --> C --> D | Photon | Light source count, placement, surface BRDF | Normal estimation under light position error | Estimate light directions, surface albedo, BRDF | Correct light calibration, inter-reflection |
| 95 | `flash_lidar` | Flash LiDAR | P --> D | Photon | Laser power, APD array, range gate | Depth map under background noise, multi-return | Estimate background rate, reflectivity, range bias | Correct range calibration, background subtraction |

---

### 5.10 Remote Sensing (11 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 96 | `sar` | Synthetic Aperture Radar | F --> D | RF | Bandwidth, PRF, look angle, aperture length | Backprojection under motion error, autofocus | Estimate platform motion errors, phase history | Correct autofocus, motion compensation |
| 97 | `sonar` | Sonar Imaging | P --> D | Acoustic | Transducer array, frequency, beamforming | DAS beamforming under sound speed error, multipath | Estimate sound speed, multipath structure | Correct sound speed, suppress multipath |
| 98 | `hyperspectral_remote` | Hyperspectral Remote Sensing | M --> W --> Sigma --> D | Photon | Spectral range, spatial resolution, push-broom vs snapshot | Unmixing under atmospheric correction error, smile/keystone | Estimate smile/keystone, atmospheric parameters | Correct spectral distortion, atmosphere |
| 99 | `multispectral_sat` | Multispectral Satellite | M --> Sigma --> D | Photon | Band selection, spatial resolution, orbit | Pan-sharpening under co-registration error, MTF difference | Estimate band-to-band registration, MTF per band | Correct registration, MTF matching |
| 100 | `gpr` | Ground-Penetrating Radar | P --> D | RF | Antenna frequency, scan spacing, time window | Migration under velocity model error, clutter | Estimate permittivity profile, clutter model | Correct velocity model, clutter suppression |
| 101 | `weather_radar` | Weather / Doppler Radar | P --> R --> D | RF | Wavelength, scan strategy, PRF, dual-pol | Reflectivity estimation under ground clutter, attenuation | Estimate clutter map, attenuation path | Correct clutter filter, attenuation |
| 102 | `radio_interferometry` | Radio Interferometry (VLBI) | F --> S --> D | RF | Baseline configuration, bandwidth, integration time | CLEAN / MEM under baseline error, atmospheric phase | Estimate baseline errors, atmospheric phase | Correct baseline, atmospheric phase |
| 103 | `passive_microwave` | Passive Microwave Radiometry | Sigma --> D | RF | Frequency, spatial resolution, integration time | Deconvolution under antenna pattern error | Estimate antenna pattern, gain calibration | Correct antenna pattern, radiometric cal |
| 104 | `insar` | InSAR (Interferometric SAR) | F --> S --> D | RF | Baseline, temporal separation, coherence | Phase unwrapping under atmospheric delay, decorrelation | Estimate atmospheric phase screen, coherence map | Correct atmospheric delay, improve coherence |
| 105 | `polsar` | Polarimetric SAR | F --> M --> D | RF | Polarization modes, calibration targets | Polarimetric decomposition under cross-pol leakage | Estimate cross-pol isolation, channel imbalance | Correct polarimetric calibration, leakage |
| 106 | `ocean_color` | Ocean Color Remote Sensing | M --> Sigma --> D | Photon | Spectral bands, spatial resolution, sun glint avoidance | Atmospheric correction under aerosol model error | Estimate aerosol optical depth, water-leaving radiance | Correct atmospheric path, sun glint removal |

---

### 5.11 Industrial Inspection (10 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 107 | `industrial_ct` | Industrial X-ray CT | Pi --> D | X-ray | kV/mA, geometry, magnification, voxel size | FBP/iterative under scatter, beam hardening, ring artifacts | Estimate center offset, ring sources, scatter fraction | Correct geometry, scatter, beam hardening |
| 108 | `xray_ndt` | X-ray NDT (Radiography) | Pi --> D | X-ray | Source type, film/DR detector, exposure chart | Enhancement under scatter, geometric unsharpness | Estimate SDD, source size, scatter buildup | Correct scatter, magnification, contrast |
| 109 | `ultrasonic_phased_array` | Ultrasonic Phased Array | P --> D | Acoustic | Element count, frequency, focal law, wedge angle | TFM/FMC under velocity error, coupling variation | Estimate velocity, coupling, element sensitivity | Correct velocity, element calibration |
| 110 | `eddy_current` | Eddy Current Imaging | F --> D | EM | Frequency, probe geometry, lift-off compensation | Impedance map under lift-off variation | Estimate lift-off, conductivity, probe alignment | Correct lift-off, conductivity scale |
| 111 | `active_thermography` | Active Thermography (IR) | P --> D | IR photon | Excitation type, camera NETD, frame rate | Thermal diffusivity inversion under non-uniform heating | Estimate emissivity map, heating uniformity | Correct emissivity, excitation model |
| 112 | `terahertz` | Terahertz Imaging | P --> D | THz photon | Frequency range, imaging mode, spatial resolution | Deconvolution under water vapor absorption, etalon | Estimate thickness, refractive index, absorption | Correct etalon artifacts, vapor lines |
| 113 | `machine_vision` | Machine Vision / AOI | C --> D | Photon | Lens, illumination, resolution, FOV | Defect detection under illumination non-uniformity | Estimate illumination profile, MTF, distortion | Correct flat-field, lens distortion, focus |
| 114 | `xrf_imaging` | X-ray Fluorescence Imaging | M --> R --> D | X-ray | Excitation energy, detector geometry, resolution | Element mapping under matrix effect, self-absorption | Estimate matrix composition, self-absorption | Correct matrix effects, dead time, pile-up |
| 115 | `shearography` | Shearography | M --> P --> D | Photon | Shearing amount, illumination, loading method | Phase map under decorrelation, rigid body motion | Estimate shear distance, decorrelation rate | Correct rigid body motion, phase unwrapping |
| 116 | `acoustic_microscopy` | Scanning Acoustic Microscopy | P --> D | Acoustic | Frequency (50-2000 MHz), lens geometry, coupling | C-scan under defocus, coupling variation | Estimate focal position, coupling impedance, velocity | Correct defocus, coupling normalization |

---

### 5.12 Scientific Instrumentation (12 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 117 | `xray_crystallography` | X-ray Crystallography | F --> S --> D | X-ray | Wavelength, rotation range, detector distance | Structure factor extraction under absorption, radiation damage | Estimate unit cell, space group, absorption | Correct absorption, scaling, damage |
| 118 | `saxs` | Small-Angle X-ray Scattering | R --> D | X-ray | Beam size, q-range, sample-detector distance | Desmearing under beam divergence, parasitic scatter | Estimate beam profile, background scatter | Correct beam smearing, background |
| 119 | `maldi_msi` | MALDI Mass Spec Imaging | S --> D | Ion | Laser spot size, step size, matrix application | Ion image under matrix inhomogeneity, mass drift | Estimate mass calibration drift, ion suppression | Correct mass calibration, normalize |
| 120 | `atom_probe` | Atom Probe Tomography | S --> D | Ion | Voltage/laser pulse, detection efficiency, FOV | 3D recon under trajectory aberration, local magnification | Estimate tip shape, local magnification | Correct geometry, compositional bias |
| 121 | `cryo_em` | Cryo-EM Single Particle | C --> D | Electron | Voltage, defocus range, dose, ice thickness | CTF correction + 3D refinement under beam tilt | Estimate CTF per micrograph, beam tilt, ice | Correct CTF, beam tilt, Ewald sphere |
| 122 | `neutron_tomo` | Neutron Tomography | Pi --> D | Neutron | Beam flux, collimation ratio, rotation steps | FBP under beam hardening, scattering, gamma | Estimate beam spectrum, scattering factor | Correct beam hardening, scatter, gamma |
| 123 | `proton_radiography` | Proton Radiography | Pi --> D | Proton | Beam energy, detector stack, angular acceptance | MLP recon under MCS model error | Estimate scattering model parameters, energy loss | Correct MCS model, energy calibration |
| 124 | `muon_tomo` | Muon Tomography | Pi --> D | Muon | Detector layers, angular resolution, integration time | POCA / MLP under angular resolution limit | Estimate detector alignment, angular uncertainty | Correct alignment, track fitting |
| 125 | `waxs` | Wide-Angle X-ray Scattering | R --> D | X-ray | Detector geometry, q-range, polarization correction | Azimuthal integration under detector tilt, beam center | Estimate beam center, tilt angles, flat-field | Correct detector geometry, polarization |
| 126 | `xrf_tomo` | XRF Tomography | Pi --> R --> D | X-ray | Excitation energy, rotation steps, self-absorption | Filtered backprojection under self-absorption error | Estimate attenuation map, fluorescence yield | Correct self-absorption, attenuation compensation |
| 127 | `neutron_diffraction` | Neutron Diffraction | R --> S --> D | Neutron | Wavelength, detector coverage, sample environment | Rietveld/Pawley under background, absorption, extinction | Estimate structure parameters, absorption correction | Correct extinction, absorption, background |
| 128 | `cathodoluminescence` | Cathodoluminescence | M --> R --> D | Electron | Beam energy, spectral range, collection optics | Hyperspectral mapping under drift, beam damage | Estimate spectral response, damage rate, drift | Correct drift, damage model, spectral calibration |

---

### 5.13 Broader Experimental Science (11 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 129 | `adaptive_optics` | Adaptive Optics (Astronomy) | M --> C --> D | Photon | Wavefront sensor, DM actuators, guide star | PSF recon under residual wavefront error | Estimate residual wavefront, Cn2 profile | Correct AO loop, turbulence model |
| 130 | `seismic_tomo` | Seismic Tomography | P --> D | Seismic | Station array, frequency band, ray geometry | Travel-time / FWI under velocity model error | Estimate velocity structure, source locations | Correct velocity model, relocate sources |
| 131 | `gravitational_wave` | Gravitational Wave Detection | P --> Sigma --> D | Gravitational | Arm length, laser power, mirror quality | Matched filter under noise non-stationarity, glitches | Estimate noise PSD, glitch morphology | Correct calibration, glitch subtraction |
| 132 | `particle_calorimetry` | Particle Calorimetry | R --> Sigma --> D | Particle | Absorber/scintillator layers, granularity | Energy/position recon under inter-calibration error | Estimate cell calibration, non-linearity curve | Correct inter-calibration, non-linearity |
| 133 | `radio_astronomy` | Radio Aperture Synthesis | F --> S --> D | RF | Antenna configuration, bandwidth, uv-coverage | CLEAN / MEM under baseline phase error, RFI | Estimate antenna gains, phase offsets, RFI | Correct antenna calibration, RFI excision |
| 134 | `acoustic_emission` | Acoustic Emission Testing | P --> S --> D | Acoustic | Sensor placement, frequency range, threshold | Source localization under velocity anisotropy | Estimate wave velocity, coupling, source type | Correct velocity model, localization |
| 135 | `magnetic_particle` | Magnetic Particle Imaging | M --> F --> D | Magnetic | FFP trajectory, drive field, selection field | System function inversion under relaxation effects | Estimate system function, relaxation params | Correct system function, resolution |
| 136 | `impedance_tomo` | Electrical Impedance Tomography | M --> D | Electric | Electrode count, current pattern, protocol | Gauss-Newton under contact impedance error | Estimate contact impedance, electrode positions | Correct contact impedance, electrode model |
| 137 | `fwi` | Full-Waveform Inversion | P --> D | Seismic/Acoustic | Source array, receiver array, frequency band | Adjoint-state inversion under starting model error | Estimate velocity model, attenuation, anisotropy | Correct starting model, cycle-skip mitigation |
| 138 | `ocean_acoustic_tomo` | Ocean Acoustic Tomography | P --> D | Acoustic | Source-receiver transects, frequency, ray paths | Travel-time inversion under sound speed profile error | Estimate sound speed profile, current velocity | Correct sound speed, ray bending model |
| 139 | `bioluminescence_tomo` | Bioluminescence Tomography | Src --> R,P --> D | Photon | Source depth, wavelength, tissue optics | Diffusion model inversion under optical property error | Estimate source location, tissue absorption/scatter | Correct optical model, source localization |

---

### 5.14 Spectroscopy & Spectral Imaging (8 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 140 | `raman_imaging` | Raman Imaging | M --> R --> D | Photon | Excitation wavelength, grating, integration time | Spectral unmixing under fluorescence background, cosmic rays | Estimate background, peak positions, linewidths | Correct fluorescence baseline, cosmic ray removal |
| 141 | `cars` | CARS Microscopy | M --> R --> D | Photon | Pump/Stokes wavelengths, bandwidth, delay | CARS recon under non-resonant background | Estimate non-resonant background, spectral phase | Correct non-resonant background, phase retrieval |
| 142 | `srs` | Stimulated Raman Scattering | M --> R --> D | Photon | Modulation frequency, pump/Stokes power, lock-in | SRS imaging under photothermal artifact, cross-phase mod | Estimate photothermal contribution, XPM artifacts | Correct photothermal, XPM background |
| 143 | `ftir_imaging` | FTIR Imaging | M --> Sigma --> D | IR photon | Spectral range, resolution, interferometer type | Interferogram processing under apodization, phase error | Estimate phase error, wavenumber calibration | Correct phase, apodization, atmospheric absorption |
| 144 | `libs` | LIBS Imaging | M --> R --> D | Photon | Laser energy, gate delay, spot size, spectrometer | Element quantification under matrix effect, self-absorption | Estimate plasma temperature, electron density | Correct matrix effects, self-absorption |
| 145 | `brillouin` | Brillouin Microscopy | M --> R --> D | Photon | VIPA etalon, extinction, NA, integration time | Brillouin shift extraction under elastic background | Estimate elastic leakage, free spectral range | Correct elastic contamination, FSR calibration |
| 146 | `sims` | SIMS Imaging | S --> D | Ion | Primary ion species, energy, mass analyzer | Mass image under dead time, mass interference | Estimate dead time, isotope ratios, matrix effect | Correct dead time, mass calibration, matrix |
| 147 | `desi` | DESI Mass Spec Imaging | S --> D | Ion | Spray solvent, voltage, spatial resolution | Mass image under signal variation, matrix effects | Estimate ion suppression, spatial offset | Correct spray normalization, co-registration |

---

### 5.15 Ultrafast Imaging (4 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 148 | `streak_camera` | Streak Camera | M --> Sigma --> D | Photon | Sweep speed, slit width, trigger jitter | Spatiotemporal recon under sweep nonlinearity, jitter | Estimate sweep function, trigger jitter, flatfield | Correct sweep nonlinearity, jitter compensation |
| 149 | `pump_probe` | Pump-Probe Microscopy | M --> R --> D | Photon | Pump/probe wavelengths, delay range, repetition rate | Transient absorption recon under chirp, scatter | Estimate chirp function, coherent artifact, scatter | Correct chirp, coherent artifact subtraction |
| 150 | `cup` | Compressed Ultrafast Photography | M --> Sigma --> D | Photon | Streak speed, DMD pattern, spatial encoding | CUP reconstruction under streak nonlinearity, DMD error | Estimate streak function, DMD misalignment | Correct streak, DMD calibration |
| 151 | `xfel_sfx` | XFEL Serial Crystallography | M --> R --> D | X-ray | Pulse energy, jet speed, hit rate, detector geometry | Indexing + merging under geometry error, partiality | Estimate detector geometry, beam center, partiality | Correct geometry, partiality model, scaling |

---

### 5.16 Quantum Imaging (3 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 152 | `ghost_imaging` | Ghost Imaging | M --> Sigma --> D | Photon | Bucket detector, spatial patterns, photon rate | Correlation reconstruction under accidental coincidences | Estimate accidental rate, visibility, detection efficiency | Correct accidentals, improve SNR via optimal patterns |
| 153 | `quantum_illumination` | Quantum Illumination | M --> R --> D | Photon | Signal-idler source, entanglement quality, detector | Target detection under background thermal noise | Estimate entanglement quality, background level | Correct noise model, optimize detection threshold |
| 154 | `entangled_photon` | Entangled Photon Microscopy | M --> R --> D | Photon | Photon pair source, coincidence window, NA | Coincidence image under accidentals, dark counts | Estimate pair rate, accidental fraction, timing jitter | Correct accidentals, timing calibration |

---

### 5.17 Multi-Modal Fusion (6 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 155 | `pet_ct` | PET/CT Fusion | Pi --> D (CT) + Pi --> D (PET) --> Fusion | X-ray + Gamma | PET/CT geometry, gantry offset, attenuation protocol | Joint recon under PET-CT misregistration, attenuation error | Estimate spatial misregistration, attenuation map error | Correct co-registration, attenuation map |
| 156 | `pet_mr` | PET/MR Fusion | Pi --> D (PET) + M --> F --> S --> D (MR) --> Fusion | Gamma + RF | Simultaneous vs sequential, attenuation from MR | Joint recon under MR-based attenuation error, susceptibility | Estimate attenuation map, geometric distortion | Correct MR-AC, reduce susceptibility artifacts |
| 157 | `spect_ct` | SPECT/CT Fusion | Pi --> D (SPECT) + Pi --> D (CT) --> Fusion | Gamma + X-ray | Collimator-CT geometry, energy windows, attenuation | Joint recon under SPECT-CT misregistration | Estimate registration offset, CT HU to mu conversion | Correct co-registration, attenuation |
| 158 | `us_mri` | US/MRI Fusion | P --> D (US) + M --> F --> S --> D (MR) --> Fusion | Acoustic + RF | Registration method, US probe tracking, MR sequence | Fused guidance under probe drift, tissue deformation | Estimate probe position drift, tissue deformation field | Correct registration, deformation compensation |
| 159 | `ct_fluorescence` | CT + Fluorescence (FLIT) | Pi --> D (CT) + M --> R,P --> D (FLI) --> Fusion | X-ray + Photon | CT structural prior, fluorophore, optical model | Joint recon under fluence error, tissue heterogeneity | Estimate fluorophore distribution, optical properties | Correct fluence, structural prior alignment |
| 160 | `clem` | Correlative Light-Electron Microscopy | C --> D (LM) + C --> D (EM) --> Fusion | Photon + Electron | Sample prep, fiducials, overlay registration | Overlay under shrinkage, distortion, scale difference | Estimate scale factor, rotation, nonlinear distortion | Correct registration, compensate shrinkage |

---

### 5.18 Scanning Probe Microscopy (4 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 161 | `afm` | Atomic Force Microscopy | S --> D | Mechanical | Cantilever spring constant, scan rate, setpoint | Topography under piezo creep, thermal drift, tip convolution | Estimate piezo coefficients, drift rate, tip shape | Correct piezo nonlinearity, drift, deconvolve tip |
| 162 | `stm` | Scanning Tunneling Microscopy | S --> D | Electron | Bias voltage, tunneling current setpoint, scan speed | Topography/LDOS under piezo hysteresis, thermal drift | Estimate piezo coefficients, tip electronic state | Correct piezo, flatten background, normalize LDOS |
| 163 | `nsom` | Near-field Scanning Optical Microscopy | M --> C --> D | Photon | Aperture size, feedback, wavelength, illumination mode | Near-field recon under topographic artifact, coupling | Estimate topographic crosstalk, far-field background | Correct topographic artifact, background subtraction |
| 164 | `mfm` | Magnetic Force Microscopy | S --> M --> D | Magnetic | Lift height, tip magnetization, scan rate | Magnetic recon under topographic crosstalk, tip degradation | Estimate lift height variation, tip moment, crosstalk | Correct lift height, topographic subtraction |

---

### 5.19 Astronomy & Space Imaging (4 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 165 | `coronagraphy` | Coronagraphy | M --> P --> D | Photon | Coronagraph type (Lyot/vortex), IWA, contrast ratio | Post-processing under speckle residuals, wind-driven halo | Estimate speckle field, quasi-static aberration | Correct speckle subtraction (ADI/SDI), wavefront |
| 166 | `lucky_imaging` | Lucky Imaging | M --> C --> D | Photon | Frame rate, selection percentage, registration method | Shift-and-add under anisoplanatism, variable seeing | Estimate Fried parameter, isoplanatic angle, tip-tilt | Correct tip-tilt, deconvolve residual PSF |
| 167 | `eht_imaging` | Event Horizon Telescope Imaging | F --> S --> D | RF (mm-wave) | Baseline coverage, bandwidth, atmospheric opacity | CLEAN/RML under atmospheric phase, sparse uv-coverage | Estimate station gains, atmospheric phase, bandpass | Correct atmospheric phase, amplitude calibration |
| 168 | `solar_imaging` | Solar Imaging (EUV/X-ray) | M --> P --> D | Photon/EUV | Wavelength channels, cadence, pointing stability | DEM reconstruction under PSF degradation, stray light | Estimate PSF degradation curve, stray light model | Correct PSF evolution, stray light subtraction |

---

## 6. Summary: Modality Count by Category

| Category | Count | Medical Physicist Relevance |
|----------|:-----:|---|
| Microscopy | 24 | Diagnostic imaging (pathology), nuclear medicine (autoradiography) |
| Compressive Imaging | 4 | Spectral tissue imaging, dose-efficient acquisition |
| Medical Imaging | 37 | **All 4 roles directly** |
| Coherent Imaging | 5 | Phase-contrast imaging, crystallography |
| Computational Photography | 5 | Endoscopic imaging, event-driven acquisition |
| Computational Optics | 2 | Light-field microscopy for pathology |
| Neural Rendering | 2 | Surgical planning, 3D anatomy |
| Electron Microscopy | 11 | Materials for devices, radiobiology, cryo-EM |
| Depth Imaging | 5 | Surface-guided radiation therapy |
| Remote Sensing | 11 | Environmental monitoring, dose mapping |
| Industrial Inspection | 10 | QA of medical devices, accelerator components |
| Scientific Instrumentation | 12 | Proton/neutron therapy verification, synchrotron |
| Broader Experimental Science | 11 | MPI (tracers), EIT (lung monitoring), seismic |
| Spectroscopy & Spectral Imaging | 8 | Tissue characterization, molecular imaging |
| Ultrafast Imaging | 4 | Radiation dynamics, XFEL crystallography |
| Quantum Imaging | 3 | Low-dose quantum-enhanced diagnostics |
| Multi-Modal Fusion | 6 | **PET/CT, PET/MR, SPECT/CT -- all 4 roles** |
| Scanning Probe Microscopy | 4 | Dosimeter characterization, materials QA |
| Astronomy & Space Imaging | 4 | AO technology transfer, EUV/X-ray detection |
| **Total** | **168** | |

---

## 7. Medical Physicist Applications: 4 Benchmarks x 4 Roles

### 7.1 Therapeutic Medical Physicist (Radiation Therapy)

**Primary modalities**: CT, CBCT, MRI, PET, portal imaging (EPID), proton therapy imaging, brachytherapy imaging, PET/CT, PET/MR, MR elastography.

| Benchmark | Application |
|-----------|------------|
| **B1** (Design) | Design imaging protocols for treatment planning: "4D-CT for lung SBRT, 0.5mm, respiratory gating" --> Spec. Design QA phantom protocols for commissioning. |
| **B2** (Simulate) | Simulate imaging chain from Spec: does the CT protocol produce sufficient contrast for tumor delineation? Feedback: "increase mA" or "add iterative recon to maintain quality at reduced dose." |
| **B3** (Identify) | Reverse-engineer system params from QA phantoms: given CBCT of Catphan, identify actual geometry (SDD, gantry flex, detector offset) vs nominal. True-Spec = manufacturer calibration. |
| **B4** (Correct) | Correct CBCT errors for dose calculation: given daily CBCT + nominal Spec, correct gantry sag, flex, ring artifacts. Feedback: "detector offset drifted 0.3mm -- recommend recalibration." |

### 7.2 Diagnostic Imaging Medical Physicist

**Primary modalities**: CT, spectral CT, MRI, MR fingerprinting, X-ray, fluoroscopy, mammography, digital breast tomosynthesis, ultrasound, IVUS, DEXA, angiography, OCT, fundus, endoscopy, confocal endomicroscopy.

| Benchmark | Application |
|-----------|------------|
| **B1** (Design) | Design protocols: "low-dose chest CT, CTDI_vol < 3 mGy" --> Spec. Design QA programs: phantom tests, acceptance criteria, testing frequency. |
| **B2** (Simulate) | Predict image quality from Spec before clinical use. Simulate: reduce dose 30%, what happens to low-contrast detectability? Feedback: "protocol exceeds DRL -- reduce mAs 15%." |
| **B3** (Identify) | Determine system params from ACR phantom images: given MRI data, estimate coil sensitivity, gradient nonlinearity, distortion. Compare vs True-Spec. Essential for accreditation. |
| **B4** (Correct) | Correct system imperfections: given MRI + Spec, correct distortion, non-uniformity, ghosting. Feedback: "coil #3 sensitivity down 12% -- recommend service." |

### 7.3 Nuclear Medicine Medical Physicist

**Primary modalities**: PET, SPECT, gamma camera, PET/CT, PET/MR, SPECT/CT, radionuclide therapy dosimetry, MPI.

| Benchmark | Application |
|-----------|------------|
| **B1** (Design) | Design protocols: "FDG-PET/CT for lymphoma, 3-min beds, TOF-OSEM, 4mm target resolution" --> Spec. Design therapy dosimetry protocols. |
| **B2** (Simulate) | Simulate PET/SPECT chain: predict NECR, scatter fraction, spatial resolution. Feedback: "time-per-bed produces SUV CV > 15% -- increase 30s." |
| **B3** (Identify) | Determine params from NEMA phantom: estimate resolution, scatter fraction, sensitivity, count rate. Compare vs True-Spec. For ACR accreditation and clinical trials. |
| **B4** (Correct) | Correct quantitative accuracy: given patient data + Spec, correct attenuation, scatter, normalization, randoms, dead time. Feedback: "block #47 normalization deviates 5% -- recalibrate PMT." |

### 7.4 Health Physics (Radiation Protection)

**Primary modalities**: Personnel dosimeters, area monitors, contamination instruments, environmental monitors, whole-body counters.

| Benchmark | Application |
|-----------|------------|
| **B1** (Design) | Design monitoring systems: "personnel dosimetry for IR suite, Hp(10) and Hp(0.07), monthly" --> Spec. Design shielding: "PET/CT suite barriers, 20 patients/day." |
| **B2** (Simulate) | Simulate dose distributions from Spec: predict dose rates at all occupied locations. Feedback: "shielding produces 0.12 mSv/wk in adjacent office -- add 0.5mm Pb to wall B." |
| **B3** (Identify) | Determine actual radiation environment from surveys: given area monitor data, reconstruct source distribution, shielding effectiveness. Compare vs True-Spec (design documents). |
| **B4** (Correct) | Correct dose estimates: given survey data + Spec, correct for occupancy, workload changes. Feedback: "measured 2x higher than predicted -- likely backscatter from new equipment." |

---

## 8. Analysis: Do the 4 Benchmarks Cover the 4-Level Targeting System?

### 8.1 Coverage Matrix

| PWM Component | B1 (Design) | B2 (Simulate) | B3 (Identify) | B4 (Correct) | Covered? |
|---------------|:-:|:-:|:-:|:-:|:-:|
| **Track 1: Correct** | | Scenario I-II | | **PRIMARY** | YES |
| **Track 2: Diagnose** | | Gate attribution | Gate identification | Feedback | YES |
| **Track 3: No-GT** | | | **PRIMARY** | When GT unavailable | YES |
| **Track 4: Design** | **PRIMARY** | Design validation | | | YES |
| **4-Scenario Protocol** | Design scenarios | Run Sc. I, II | Inform Sc. III | Run Sc. III, IV | YES |
| **Triad Gate 1** (Sampling) | Design sampling | Simulate limits | Identify sampling | Diagnose sampling | YES |
| **Triad Gate 2** (Noise) | Design noise budget | Simulate noise | Identify noise | Diagnose noise | YES |
| **Triad Gate 3** (Mismatch) | Design tolerance | Simulate mismatch | Identify params | **Correct mismatch** | YES |
| **Red Team injection** | Novel requirements | Novel mismatch | Unknown configs | Compound failures | YES |
| **Anti-Goodhart** | Pareto optimality | Prospective fidelity | Parameter accuracy | Recovery ratio rho | YES |
| **RoIC** | Design for calibratability | Simulation cost | Identification cost | dB/GPU-hr | YES |
| **Uncertainty** | Robustness margin | Noise propagation | Parameter CIs | Calibrated CIs | YES |

### 8.2 The Two Flowcharts Form a Closed Loop

```
     Flowchart 1 (Spec-First)          Flowchart 2 (Data-First)
    ┌─────────┐    ┌─────────┐       ┌─────────┐    ┌─────────┐
    │   B1    │───>│   B2    │       │   B3    │───>│   B4    │
    │ Design  │    │Simulate │       │Identify │    │ Correct │
    └────┬────┘    └────┬────┘       └────┬────┘    └────┬────┘
         │              │                  │              │
         │              │    feedback      │              │
         │              └─────────────────>│              │
         │                                 │              │
         │              redesign           │              │
         │<────────────────────────────────┘              │
         │                                                │
         │              Spec for correction               │
         └───────────────────────────────────────────────>│
                                                          │
                        improved Spec                     │
         <────────────────────────────────────────────────┘
```

1. **B1** designs a Spec from requirements
2. **B2** validates the Spec by simulating forward and reconstructing; feedback reveals which parameters matter
3. **B3** reverse-engineers the Spec from real data (closing the sim-to-real gap)
4. **B4** fixes the Spec when reality deviates, reconstructs, and produces feedback that feeds back into B1

The two flowcharts together implement the **Design-Make-Test-Iterate** cycle.

### 8.3 Verdict

**Yes, the four benchmarks fully cover the 4-level targeting system.** The two-flowchart architecture provides:

- **Flowchart 1 (B1+B2)** covers the **forward direction**: design and validate before building.
- **Flowchart 2 (B3+B4)** covers the **inverse direction**: identify and correct after building.
- **Independence** ensures each benchmark can be used as a standalone competition, evaluation, or QA tool.
- **Cross-flowchart connections** close the loop for continuous improvement.

The four benchmarks are **necessary and complementary**:
- Without B1: no design evaluation.
- Without B2: no forward-model validation; the chain from Spec to measurement is untested.
- Without B3: no system identification; the chain from measurement to Spec is untested.
- Without B4: no correction evaluation; the core ISA capability is untested.

---

## 9. Implementation: True-Spec Schema

### 9.1 True-Spec Structure

```yaml
# true_spec.yaml -- hidden from contestants, used by LIP Arena for scoring
version: "1.0"
modality: cassi
benchmark: B4

system:
  primitives:
    - {type: M, id: coded_mask, params: {density: 0.5, shift_x: 1.47, shift_y: -0.23, rotation_deg: 0.31}}
    - {type: W, id: spectral_dispersion, params: {slope: 2.01, offset: 0.04}}
    - {type: Sigma, id: spectral_sum, params: {}}
    - {type: D, id: cmos_detector, params: {gain: 1.02, read_noise: 5.1, dark_current: 0.3}}
  noise:
    type: poisson_gaussian
    params: {shot: true, read_sigma: 5.1, dark_rate: 0.3}

mismatch_ranges:  # disclosed to contestants
  shift_x: {min: -3.0, max: 3.0}
  shift_y: {min: -3.0, max: 3.0}
  rotation_deg: {min: -2.0, max: 2.0}
  dispersion_slope: {min: 1.5, max: 2.5}
  gain: {min: 0.9, max: 1.1}

recommendations:  # for judging B4 feedback quality
  - {priority: 1, action: "recalibrate mask alignment", expected_gain_dB: 5.0}
  - {priority: 2, action: "re-measure dispersion curve", expected_gain_dB: 1.2}
  - {priority: 3, action: "reduce read noise via cooling", expected_gain_dB: 0.3}
```

### 9.2 Benchmark Pack Structure

```
benchmark_pack/
  modality.yaml
  B1/
    prompt.txt
    original_spec.yaml       # optional
    true_spec.yaml            # hidden (for B1: ideal design reference)
  B2/
    spec.yaml                 # input (from B1 or user-provided)
    true_spec.yaml            # hidden (injected mismatch params)
  B3/
    dataset/
      y.npy                   # measurement data
      H_nominal.npz           # nominal system matrix
      metadata.json
    prompt.txt
    original_spec.yaml        # optional
    true_spec.yaml            # hidden (exact system parameters)
  B4/
    dataset/
      y.npy
      H_nominal.npz
      metadata.json
    spec.yaml                 # imperfect (from B3 or user-provided)
    true_spec.yaml            # hidden (exact params + recommendations)
```

---

## 10. Cross-Reference: Categories x Medical Physicist Roles

| Category | Therapeutic MP | Diagnostic MP | Nuclear Med MP | Health Physics |
|----------|:-:|:-:|:-:|:-:|
| Microscopy | Radiobiology | Pathology QA | Autoradiography | Bioassay |
| Compressive | Dose-efficient imaging | Spectral tissue | Compressed PET/SPECT | |
| Medical (X-ray) | **CBCT, portal, tomo** | **CT, spectral CT, mammo, fluoro, DBT** | PET/CT atten. | **Shielding** |
| Medical (MRI) | **MR-guided RT, MRE** | **Clinical MRI QA, MRF** | PET/MR | RF safety |
| Medical (Nuclear) | **PET targeting** | | **PET, SPECT, dosimetry** | Contamination |
| Medical (Ultrasound) | **US-guided brachy** | **Clinical US QA, IVUS, CEUS** | | |
| Clinical Optics | | **Fundus, OCT, confocal endo** | | |
| Coherent Imaging | Talbot-Lau phase contrast | Phase imaging | | |
| Computational Photography | | Coded exposure, event camera | | |
| Computational Optics | | Light-field pathology | | |
| Neural Rendering | Surgical planning | 3D anatomy | | |
| Electron Microscopy | | | | Dosimeter char. |
| Depth Imaging | **Surface-guided RT** | | | |
| Remote Sensing | | | Environmental | **Area monitoring** |
| Industrial Inspection | Accelerator QA | Device QA | Source QC | **Shielding survey** |
| Scientific Instr. | Proton/neutron verify | Synchrotron imaging | | |
| Broader Exp. Science | MPI therapy | EIT ventilation | | |
| Spectroscopy & Spectral | Tissue characterization | Raman pathology | Radiolabel analysis | |
| Ultrafast Imaging | Radiation dynamics | | | |
| Quantum Imaging | Low-dose quantum | | | |
| Multi-Modal Fusion | **PET/CT, PET/MR** | **US/MRI, FLIT** | **PET/CT, SPECT/CT** | |
| Scanning Probe | | | | Dosimeter materials |
| Astronomy & Space | | | | AO/EUV transfer |

---

## 11. References

1. PWM Flagship Paper -- [Typed Primitives and OperatorGraph IR](https://github.com/integritynoble/Physics_World_Model/blob/master/papers/pwm_flagship/main.pdf)
2. PWM Targeting System -- [`docs/targeting_system.md`](targeting_system.md)
3. PWM Imaging Modality Registry -- [`docs/imaging_modalities.md`](imaging_modalities.md)
4. PWM Operator Mode -- [`docs/operator_mode.md`](operator_mode.md)
5. PWM Canonical Primitives -- [`docs/plan_canonical_primitives.md`](plan_canonical_primitives.md)
6. PWM Purpose (ISA) -- [`docs/purpose.md`](purpose.md)
7. PWM Per-Modality Benchmarks -- [`docs/pwm_modality_benchmarks_detailed.md`](pwm_modality_benchmarks_detailed.md)
8. SolveEverything Framework -- [https://solveeverything.org/](https://solveeverything.org/)
9. AAPM Task Groups -- TG-142, TG-66, TG-174
10. CAMPEP Standards -- Medical physicist subspecialty definitions
