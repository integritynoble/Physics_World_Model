# PWM 4-Benchmark Targeting System: Universal Imaging Modality Coverage

> **Document scope**: Defines four benchmarks for every imaging modality in PWM,
> maps them to the PWM 4-level targeting system (Tracks A-D / Phases A-D),
> and examines how they serve the four professional roles of medical physicists.
>
> **Reference**: [PWM Flagship Paper — Typed Primitives](https://github.com/integritynoble/Physics_World_Model/blob/master/papers/pwm_flagship/main.pdf)

---

## 1. The 4-Benchmark Framework

Every imaging modality in PWM is evaluated through **four benchmarks** that progressively test the full cycle from system design to autonomous correction. Each benchmark has a **Spec** (the machine-readable description of how a dataset is obtained) and, where applicable, a **True-Spec** (the hidden ground-truth parameters used to judge quality).

A **Spec** is constructed from the **typed primitives** defined in the flagship paper:

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

### Benchmark 1 (B1): Prompt + Original-Spec --> Spec  (DESIGN)

| Field | Description |
|-------|-------------|
| **Input** | Natural-language prompt describing imaging requirements (e.g., "hyperspectral imaging of tissue at 450-650 nm, 28 bands, snapshot acquisition") + optional Original-Spec from a previous round |
| **Output** | A complete Spec: DAG of typed primitives, system elements (lens, mask, detector), acquisition parameters, noise model, mismatch tolerance envelope |
| **Evaluation** | (a) Physical validity -- does the Spec define a realizable imaging system? (b) Constraint satisfaction -- does it meet the stated requirements? (c) Robustness margin -- how much mismatch can it tolerate before degradation? (d) Calibration cost -- how many GPU-hours to correct from tolerance-edge mismatch? |
| **True-Spec** | Not applicable (generative benchmark); evaluated by forward-simulation feasibility and Pareto optimality |
| **PWM Track** | **Track 4: Design** -- requirements to robust OperatorGraph |

**What it tests**: Can the system (or contestant) design an imaging system from a text description, grounding the design in the typed-primitive vocabulary?

---

### Benchmark 2 (B2): Spec --> Reconstructions  (FORWARD + RECONSTRUCT)

| Field | Description |
|-------|-------------|
| **Input** | A complete Spec (from B1 or provided) |
| **Output** | (a) Simulated measurement `y` with realistic noise and mismatch injected per the Spec's tolerance envelope, (b) Reconstruction `x_hat` using declared solver, (c) Feedback report on how to improve the system (which Triad gate is binding, what mismatch dominates) |
| **Evaluation** | (a) Forward-model fidelity -- does the Spec-derived operator match reference physics (epsilon_tier2 < 0.01)? (b) Reconstruction quality under nominal conditions (Scenario I), (c) Reconstruction quality under mismatch (Scenario II), (d) Quality of feedback -- are suggested improvements actionable? |
| **True-Spec** | The Spec itself is ground truth for forward simulation; the injected mismatch parameters are known |
| **PWM Track** | Combines **Track 1: Correct** (Scenario I-II baseline) and **Track 2: Diagnose** (feedback/attribution) |

**What it tests**: Given a complete system description, can you faithfully simulate it, reconstruct from the simulated data, and diagnose what limits performance?

---

### Benchmark 3 (B3): Dataset + Prompt + Original-Spec --> Spec  (SYSTEM IDENTIFICATION)

| Field | Description |
|-------|-------------|
| **Input** | Real measurement dataset `y` (measurement matrix, calibration data, metadata) + natural-language prompt + optional Original-Spec |
| **Output** | An inferred Spec that explains how the dataset was obtained -- the DAG of primitives, estimated system parameters, noise model, mismatch characterization |
| **Evaluation** | Compare inferred Spec against hidden True-Spec: (a) Parameter RMSE for all mismatch parameters, (b) Correct identification of the primitive DAG, (c) Correct noise model identification, (d) Uncertainty calibration (do declared CIs contain truth?) |
| **True-Spec** | Hidden ground truth containing all exact parameters: mask rotation degree, shift offsets, noise level, gain drift, dispersion coefficients, etc. Contestants know the *range* of mismatch but not the exact values |
| **PWM Track** | **Track 3: No-GT** (system identification from data without ground truth reconstruction) + **Track 2: Diagnose** |

**What it tests**: Given real data and a rough description, can you reverse-engineer the imaging system -- identifying what primitives were used and what their parameters are?

---

### Benchmark 4 (B4): Dataset + Spec --> Correction + Reconstruction + Feedback  (CORRECT + DIAGNOSE)

| Field | Description |
|-------|-------------|
| **Input** | Real measurement dataset `y` + an imperfect Spec (containing mismatch errors within declared tolerance envelope) |
| **Output** | (a) Corrected Spec (calibrated operator parameters), (b) Reconstruction `x_hat` using corrected operator, (c) Feedback: TriadReport (dominant gate, evidence, confidence), suggested improvements to Spec and experimental system |
| **Evaluation** | Compare against hidden True-Spec: (a) Recovery ratio rho = (PSNR_III - PSNR_II) / (PSNR_I - PSNR_II), (b) Parameter recovery RMSE, (c) TriadReport accuracy (correct dominant gate?), (d) Feedback quality -- would the suggestions actually improve the real system? (e) Compute efficiency RoIC (dB/GPU-hr) |
| **True-Spec** | Hidden ground truth; also contains recommendations for improving the real experimental system, used to judge feedback quality |
| **PWM Track** | **Track 1: Correct** (primary) + **Track 2: Diagnose** (feedback) + **Track 3: No-GT** (when ground truth reconstruction unavailable) |

**What it tests**: The full ISA pipeline -- can you fix what's wrong with the system model, reconstruct well, and provide actionable guidance for improving the physical instrument?

---

## 2. Mapping the 4 Benchmarks to the PWM 4-Level Targeting System

### 2.1 Mapping to Evaluation Tracks

| Benchmark | Track 1: Correct | Track 2: Diagnose | Track 3: No-GT | Track 4: Design |
|-----------|:-:|:-:|:-:|:-:|
| **B1** (Prompt-->Spec) | | | | **PRIMARY** |
| **B2** (Spec-->Recon) | Scenario I-II baseline | Feedback/attribution | | |
| **B3** (Data-->Spec) | | Parameter identification | **PRIMARY** | |
| **B4** (Data+Spec-->Correction) | **PRIMARY** | Feedback | When no GT available | |

**Conclusion**: The 4 benchmarks provide **complete coverage** of all 4 evaluation tracks. Each track has at least one primary benchmark, and the cross-connections ensure that diagnosis, correction, and design are tested from multiple angles.

### 2.2 Mapping to Maturation Phases

| Phase | Timeline | Benchmarks Active | Key Milestone |
|-------|----------|-------------------|---------------|
| **Phase A** (Internal) | 0-6 months | B2, B4 on 7 validated modalities | Sealed-simulator B2+B4 for CASSI, CACTI, SPC, CT, Ptychography, MRI, Lensless |
| **Phase B** (Pilot) | 6-12 months | B1-B4 on 10+ modalities | B1 and B3 added; first live-lab True-Specs; external submissions |
| **Phase C** (Full) | 12-24 months | B1-B4 on 20+ modalities | All benchmarks operational with Red Team adversarial injection |
| **Phase D** (Utility) | 24+ months | B1-B4 on 64+ modalities | Hardware-in-the-loop; B1 from natural language to validated pipeline in minutes |

### 2.3 Mapping to Maturation Levels (L0-L5)

| Level | B1 Capability | B2 Capability | B3 Capability | B4 Capability |
|-------|---------------|---------------|---------------|---------------|
| **L0** (Muddle) | No spec language | Ad-hoc simulation | Manual system ID | Manual calibration |
| **L1** (Measurable) | Spec templates exist | Metrics defined, comparable | Parameter ranges known | Recovery ratio tracked |
| **L2** (Repeatable) | SOP for spec writing | Documented forward pipelines | Documented ID procedures | Calibration SOPs codified |
| **L3** (Automated) | Auto-spec from prompt | Auto-simulation + diagnosis | Auto-identification from data | Auto-correction, rho >= 0.80 |
| **L4** (Industrialized) | Design-as-a-service | Simulation-as-a-service | Identification-as-a-service | Calibration-as-a-service |
| **L5** (Commodity) | Any system self-designs | Any system self-simulates | Any system self-identifies | Any system self-corrects |

---

## 3. Complete Modality Registry with 4-Benchmark Specifications

### 3.1 Microscopy (13 + 3 expanded = 16 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design Focus | B2 Forward/Recon Focus | B3 Identification Focus | B4 Correction Focus |
|---|-------------|-----------|---------------|---------|----------------|----------------------|------------------------|-------------------|
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

---

### 3.2 Compressive Imaging (4 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 17 | `cassi` | CASSI (Spectral) | M --> W --> Sigma --> D | Photon | Mask pattern, dispersion element, spectral range, spatial resolution | GAP-TV / MST-L under mask shift (dx,dy), rotation (theta), dispersion slope (a1), axis offset (alpha) | Estimate 5 mismatch params from measurement residuals | Correct all 5 params; recovery ratio rho validated at 85% (flagship) |
| 18 | `spc` | Single-Pixel Camera | M --> Sigma --> D | Photon | Pattern type (Hadamard/Gaussian), sampling rate, DMD resolution | FISTA-TV under gain drift (alpha) and measurement noise (sigma_y) | Estimate gain drift curve and noise level | Correct gain model; rho validated at 86% |
| 19 | `cacti` | CACTI (Temporal) | M --> Sigma --> D | Photon | Mask shift type, compression ratio, temporal resolution | GAP-TV under spatial shift, rotation, temporal clock, gain, offset | Estimate 8 mismatch params from temporal correlations | Correct mask replication errors; rho validated at 100% |
| 20 | `matrix` | Generic Matrix Sensing | M --> D | Photon | Measurement matrix design (RIP, coherence), conditioning | CG/ADMM under matrix perturbation | Estimate matrix condition number, perturbation magnitude | Correct matrix calibration errors |

**CASSI Validated Baselines** (from flagship paper):

| Solver | Gain (dB) | Recovery rho | RoIC (dB/GPU-hr) |
|--------|-----------|-------------|-------------------|
| GAP-TV | +0.76 | 85% | 0.9 |
| MST-L | +3.01 | 26% | -- |

---

### 3.3 Medical Imaging (17 + 8 expanded = 25 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 21 | `ct` | X-ray CT | Pi --> D | X-ray | Geometry (fan/parallel/cone), angles, detector count, dose | FBP/SART under center-of-rotation offset, angular offset, detector tilt, beam hardening | Estimate CoR offset, angular errors, hardening coefficients | Correct geometry; rho validated at 100%, +10.68 dB |
| 22 | `mri` | MRI | M --> F --> S --> D | Spin/RF | Coil count, trajectory (Cartesian/radial/spiral), acceleration factor | SENSE/GRAPPA under coil sensitivity error, k-space trajectory deviation, off-resonance | Estimate coil maps, trajectory errors, field map | Correct coil + trajectory; +1.75-7.14 dB gain |
| 23 | `xray_radiography` | X-ray Radiography | Pi --> D | X-ray | Source-detector distance, filtration, exposure parameters | TV-FISTA under scatter, beam hardening, detector lag | Estimate scatter fraction, hardening polynomial | Correct scatter model, apply beam hardening correction |
| 24 | `ultrasound` | Ultrasound B-mode | P --> D | Acoustic | Transducer array, frequency, focus depth, apodization | DAS beamforming under speed-of-sound error, phase aberration | Estimate sound speed profile, aberration screen | Correct aberration, adaptive beamforming |
| 25 | `pet` | PET | Pi --> D | Gamma | Crystal ring geometry, TOF resolution, attenuation correction | MLEM/OSEM under attenuation map error, scatter, randoms, normalization | Estimate attenuation correction factors, scatter fraction, normalization table | Correct attenuation map, scatter estimation, detector normalization |
| 26 | `spect` | SPECT | Pi --> D | Gamma | Collimator type, orbit, energy window, attenuation correction | MLEM with depth-dependent resolution under collimator response error, attenuation error | Estimate collimator parameters, attenuation map, center-of-rotation | Correct CoR, update collimator response model, attenuation correction |
| 27 | `fluoroscopy` | Fluoroscopy | Pi --> D | X-ray | Frame rate, dose per frame, detector (II vs flat panel) | TV-FISTA under temporal lag, scatter, geometric distortion | Estimate lag coefficient, scatter model, pincushion distortion | Correct temporal lag, flat-field, geometric distortion |
| 28 | `mammography` | Mammography | Pi --> D | X-ray | Target/filter combination, compression, detector type | TV-FISTA under scatter, heel effect, detector MTF variation | Estimate scatter-to-primary ratio, heel effect profile | Correct scatter, apply detector-specific MTF correction |
| 29 | `dexa` | DEXA | Pi --> D | X-ray | Dual energy selection, scan mode (pencil/fan), calibration phantom | Dual-energy decomposition under beam hardening, fat-lean mismatch | Estimate effective energies, calibration polynomial | Correct calibration, update decomposition coefficients |
| 30 | `cbct` | Cone-Beam CT (CBCT) | Pi --> D | X-ray | Cone angle, flat-panel geometry, rotation arc, dose | FDK under cone-beam artifacts, scatter, truncation | Estimate scatter fraction, truncation extent, detector offset | Correct scatter, extend FOV, correct ring artifacts |
| 31 | `angiography` | X-ray Angiography | Pi --> D | X-ray | Contrast injection timing, frame rate, subtraction protocol | DSA subtraction under patient motion, misregistration | Estimate motion field between mask and contrast frames | Correct motion-compensated subtraction |
| 32 | `dot` | Diffuse Optical Tomography | M --> R,P,R --> D | Photon | Source-detector layout, wavelength selection, time/frequency domain | Born approximation inversion under scattering coefficient error | Estimate absorption and scattering coefficients, boundary conditions | Correct optical properties, improve boundary model |
| 33 | `photoacoustic` | Photoacoustic | M --> P --> D | Acoustic | Transducer array, laser wavelength, fluence model | Backprojection under speed-of-sound heterogeneity, acoustic attenuation | Estimate sound speed map, Grueneisen parameter, fluence | Correct sound speed model, fluence compensation |
| 34 | `oct` | OCT | P+P --> Sigma --> D | Photon | Source bandwidth, reference arm, scan pattern, axial resolution | FFT recon under dispersion mismatch, reference arm drift | Estimate dispersion coefficients, reference arm position | Correct dispersion, compensate for reference drift |
| 35 | `fmri` | Functional MRI (BOLD) | M --> F --> S --> D | Spin/RF | TR/TE, spatial resolution, temporal resolution, EPI trajectory | SENSE + GLM under geometric distortion, signal dropout, physiological noise | Estimate field map, distortion, motion parameters | Correct geometric distortion, motion, physiological noise |
| 36 | `mrs` | MR Spectroscopy | M --> F --> S --> D | Spin/RF | Voxel localization, spectral bandwidth, water suppression | LCModel fitting under lineshape distortion, baseline error, eddy currents | Estimate lineshape, eddy current phase, residual water | Correct lineshape, eddy current correction, baseline |
| 37 | `diffusion_mri` | Diffusion MRI (DTI) | M --> F --> S --> D | Spin/RF | b-values, gradient directions, spatial resolution, eddy currents | WLS tensor fitting under gradient nonlinearity, eddy current distortion | Estimate gradient tables, eddy current coefficients, motion | Correct gradient nonlinearity, eddy current distortion |
| 38 | `doppler_ultrasound` | Doppler Ultrasound | P --> D | Acoustic | PRF, wall filter, velocity range, angle of insonation | Autocorrelation estimator under aliasing, wall filter error, angle error | Estimate flow angle, PRF aliasing threshold, clutter level | Correct angle, anti-aliasing, clutter filter |
| 39 | `elastography` | Shear-Wave Elastography | P --> D | Acoustic | Push pulse, tracking method, shear wave frequency | TOF inversion under wave reflection, dispersion, noise | Estimate shear wave speed, attenuation, boundary effects | Correct reflection artifacts, dispersion compensation |
| 40 | `endoscopy` | Fiber Bundle Endoscopy | M --> C --> D | Photon | Fiber count, FOV, bending radius, illumination | TV-FISTA under fiber cross-talk, non-uniform transmission | Estimate fiber transmission map, geometric distortion | Correct fiber calibration, geometric distortion |
| 41 | `fundus` | Fundus Camera | C --> D | Photon | FOV, illumination wavelength, mydriasis | Richardson-Lucy under optical aberration, non-uniform illumination | Estimate aberration coefficients, illumination profile | Correct aberrations, flat-field normalization |
| 42 | `octa` | OCT Angiography | P+P --> Sigma --> D | Photon | Scan density, interscan time, decorrelation method | TV-FISTA under bulk motion, projection artifact | Estimate bulk motion, shadow artifacts | Correct bulk motion, projection artifact removal |
| 43 | `proton_therapy_img` | Proton Therapy Imaging | Pi --> D | Proton | Energy range, detector stack, range verification | Backprojection under range uncertainty, scattering | Estimate water-equivalent path length, scattering model | Correct range model, scattering compensation |
| 44 | `brachytherapy_img` | Brachytherapy Imaging | Pi --> D | Gamma/X-ray | Source geometry, applicator model, imaging protocol | TG-43/TG-186 dose with imaging verification | Estimate source position, applicator geometry from images | Correct source localization, applicator model |
| 45 | `portal_imaging` | Portal Imaging (EPID) | Pi --> D | MV X-ray | Detector geometry, gantry angle, field size | Back-projection under sag, flex, MLC position error | Estimate gantry sag, detector offset, MLC leaf positions | Correct geometric calibration, MLC model |

**CT Validated Baselines** (from flagship paper):

| Solver | Gain (dB) | Recovery rho | RoIC (dB/GPU-hr) |
|--------|-----------|-------------|-------------------|
| FBP | +10.68 | 100% | 120 |

**MRI Validated Baselines**:

| Config | Gain (dB) | Recovery rho | RoIC (dB/GPU-hr) |
|--------|-----------|-------------|-------------------|
| SENSE R=4 | +1.75-7.14 | 20% | 5790 |

---

### 3.4 Coherent Imaging (3 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 46 | `ptychography` | Ptychographic Imaging | M --> P --> D | Electron/Photon | Probe size, overlap ratio, scan pattern, coherence | ePIE under probe position error, defocus, aberration | Estimate probe positions, aberration coefficients | Correct positions; rho validated at 100%, +7.09 dB |
| 47 | `holography` | Digital Holographic Microscopy | P --> D | Photon | Reference beam angle, wavelength, off-axis vs inline | Angular spectrum under reference beam angle error, vibration | Estimate carrier frequency, reference angle, phase offset | Correct reference beam model, vibration compensation |
| 48 | `phase_retrieval` | Coherent Diffractive Imaging | P --> D | Photon/Electron | Support constraint, oversampling ratio, coherence | HIO/ER under support error, partial coherence | Estimate support boundary, coherence function | Correct support, update coherence model |

---

### 3.5 Computational Photography (2 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 49 | `lensless` | Lensless / Diffuser Camera | C --> D | Photon | Diffuser/mask type, sensor distance, PSF calibration | ADMM under PSF shift, scale drift, defocus | Estimate PSF shift, scale, defocus offset | Correct PSF model; rho validated at 78%, +3.55 dB |
| 50 | `panorama` | Panorama Multi-Focus Fusion | C --> D | Photon | Focal sweep range, number of focal planes, depth of field | Laplacian pyramid fusion under focus distance error | Estimate focal distances, aperture, depth map | Correct focal plane registration |

---

### 3.6 Computational Optics (2 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 51 | `light_field` | Light Field Imaging | C --> S --> D | Photon | Microlens array, angular resolution, spatial resolution tradeoff | Shift-and-sum / depth estimation under microlens alignment error | Estimate microlens pitch, rotation, f-number | Correct microlens calibration |
| 52 | `integral` | Integral Photography | C --> S --> D | Photon | Lens array geometry, baseline, depth range | Depth estimation under lens distortion | Estimate lens positions, distortion coefficients | Correct geometric calibration |

---

### 3.7 Neural Rendering (2 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 53 | `nerf` | Neural Radiance Fields | M --> P --> D | Photon | View count, camera placement, scene bounds, sampling density | NeRF/Instant-NGP under camera pose error, intrinsic error | Estimate camera poses, focal length, distortion | Correct camera calibration, refine poses |
| 54 | `gaussian_splatting` | 3D Gaussian Splatting | M --> P --> D | Photon | Initial point cloud, densification strategy, view selection | 3DGS optimization under SfM initialization error | Estimate point cloud quality, initialization bias | Correct initialization, re-densify |

---

### 3.8 Electron Microscopy (8 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 55 | `sem` | SEM | C --> D | Electron | Beam energy, working distance, detector type | Direct imaging under charging, drift, astigmatism | Estimate stigmation, working distance, drift rate | Correct astigmatism, drift compensation |
| 56 | `tem` | TEM | C --> D | Electron | Acceleration voltage, aperture, defocus series | CTF correction under defocus, astigmatism, beam tilt | Estimate CTF parameters (defocus, Cs, astigmatism) | Correct CTF, update aberration model |
| 57 | `electron_tomography` | Electron Tomography | Pi --> D | Electron | Tilt range, tilt increment, missing wedge | SIRT/WBP under tilt axis misalignment, magnification change | Estimate tilt axis offset, magnification variation | Correct tilt axis, compensate missing wedge |
| 58 | `stem` | STEM | S --> D | Electron | Convergence angle, detector geometry, scan pattern | Direct imaging under scan distortion, probe aberration | Estimate scan distortion, probe parameters | Correct scan calibration |
| 59 | `electron_diffraction` | 4D-STEM Diffraction | M --> P --> D | Electron | Probe size, scan step, convergence semi-angle, camera length | Ptychographic reconstruction under camera length error | Estimate camera length, beam center, rotation | Correct geometry calibration |
| 60 | `ebsd` | EBSD | R --> D | Electron | Tilt angle, step size, accelerating voltage, detector geometry | Hough indexing under pattern center error, detector distortion | Estimate pattern center (PC), detector tilt | Correct PC calibration |
| 61 | `eels` | EELS | S --> D | Electron | Energy range, dispersion, collection angle | Fourier ratio under energy drift, gain variation | Estimate energy drift, gain instability, zero-loss shift | Correct energy calibration, gain normalization |
| 62 | `electron_holography` | Electron Holography | P --> D | Electron | Biprism voltage, fringe spacing, FOV | Fourier sideband extraction under biprism drift | Estimate biprism voltage drift, fringe rotation | Correct fringe analysis parameters |

---

### 3.9 Clinical Optics (3 modalities)

Covered in Medical Imaging section (modalities 40-42: `endoscopy`, `fundus`, `octa`).

---

### 3.10 Depth Imaging (3 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 63 | `tof_camera` | Time-of-Flight Camera | P --> D | Photon/IR | Modulation frequency, integration time, multi-path | TV-FISTA under multi-path interference, phase wrap | Estimate multi-path coefficients, phase wrap count | Correct multi-path, phase unwrapping |
| 64 | `lidar` | LiDAR Scanner | P --> S --> D | Photon | Scan pattern, pulse rate, wavelength, range | Point cloud reconstruction under timing jitter, angular error | Estimate timing calibration, angular encoder error | Correct timing, angular calibration |
| 65 | `structured_light` | Structured-Light 3D | M --> C --> D | Photon | Pattern type (binary/sinusoidal/Gray code), projector-camera geometry | Phase unwrapping under projector defocus, gamma nonlinearity | Estimate gamma curve, projector-camera extrinsics | Correct gamma, geometric calibration |

---

### 3.11 Remote Sensing (2 + 6 expanded = 8 modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 66 | `sar` | Synthetic Aperture Radar | F --> D | RF | Bandwidth, PRF, look angle, aperture length | Backprojection under motion error, autofocus | Estimate platform motion errors, phase history | Correct autofocus, motion compensation |
| 67 | `sonar` | Sonar Imaging | P --> D | Acoustic | Transducer array, frequency, beamforming | DAS beamforming under sound speed error, multipath | Estimate sound speed, multipath structure | Correct sound speed model, suppress multipath |
| 68 | `hyperspectral_remote` | Hyperspectral Remote Sensing | M --> W --> Sigma --> D | Photon | Spectral range, spatial resolution, push-broom vs snapshot | Unmixing under atmospheric correction error, smile/keystone | Estimate smile/keystone distortion, atmospheric parameters | Correct spectral distortion, atmospheric compensation |
| 69 | `multispectral_sat` | Multispectral Satellite | M --> Sigma --> D | Photon | Band selection, spatial resolution, orbit parameters | Pan-sharpening / fusion under co-registration error, MTF difference | Estimate band-to-band registration, MTF per band | Correct registration, MTF matching |
| 70 | `gpr` | Ground-Penetrating Radar | P --> D | RF | Antenna frequency, scan spacing, time window | Migration/backprojection under velocity model error, clutter | Estimate dielectric permittivity profile, clutter model | Correct velocity model, clutter suppression |
| 71 | `weather_radar` | Weather / Doppler Radar | P --> R --> D | RF | Wavelength, scan strategy, PRF, dual-pol | Reflectivity / velocity estimation under ground clutter, attenuation | Estimate clutter map, attenuation path, calibration offset | Correct ground clutter filter, attenuation correction |
| 72 | `radio_interferometry` | Radio Interferometry (VLBI) | F --> S --> D | RF | Baseline configuration, bandwidth, integration time | CLEAN / MEM under baseline calibration error, atmospheric phase | Estimate baseline errors, ionospheric/tropospheric phase | Correct baseline calibration, atmospheric phase correction |
| 73 | `passive_microwave` | Passive Microwave Radiometry | Sigma --> D | RF | Frequency, spatial resolution, integration time | Deconvolution under antenna pattern error, cross-polarization | Estimate antenna pattern, gain calibration | Correct antenna pattern, radiometric calibration |

---

### 3.12 Industrial Inspection (8 new modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 74 | `industrial_ct` | Industrial X-ray CT | Pi --> D | X-ray | kV/mA, geometry, magnification, voxel size for part inspection | FBP/iterative under scatter, beam hardening, ring artifacts | Estimate center offset, ring artifact sources, scatter fraction | Correct geometry, scatter, beam hardening for dimensional metrology |
| 75 | `xray_ndt` | X-ray NDT (Radiography) | Pi --> D | X-ray | Source type (tube/isotope), film/DR detector, exposure chart | Enhancement + defect detection under scatter, geometric unsharpness | Estimate SDD, source size, scatter buildup factor | Correct scatter, geometric magnification, contrast enhancement |
| 76 | `ultrasonic_phased_array` | Ultrasonic Phased Array | P --> D | Acoustic | Element count, frequency, focal law, wedge angle | TFM/FMC under velocity error, wedge coupling variation | Estimate velocity, wedge coupling, element sensitivity | Correct velocity model, element calibration |
| 77 | `eddy_current` | Eddy Current Imaging | F --> D | EM | Frequency, probe geometry, lift-off compensation | Impedance map under lift-off variation, conductivity gradient | Estimate lift-off, conductivity profile, probe alignment | Correct lift-off, calibrate conductivity scale |
| 78 | `active_thermography` | Active Thermography (IR) | P --> D | IR photon | Excitation type (flash/lock-in/pulse), camera NETD, frame rate | Thermal diffusivity inversion under non-uniform heating, emissivity variation | Estimate emissivity map, heating uniformity, diffusivity | Correct emissivity, non-uniform excitation model |
| 79 | `terahertz` | Terahertz Imaging | P --> D | THz photon | Frequency range, imaging mode (reflection/transmission), spatial resolution | Deconvolution / tomography under water vapor absorption, etalon effects | Estimate material thickness, refractive index, absorption | Correct etalon artifacts, water vapor lines |
| 80 | `machine_vision` | Machine Vision / AOI | C --> D | Photon | Lens, illumination (backlight/ring/dome), resolution, field of view | Defect detection under illumination non-uniformity, focus variation | Estimate illumination profile, MTF, geometric distortion | Correct flat-field, lens distortion, focus map |
| 81 | `xrf_imaging` | X-ray Fluorescence Imaging | M --> R --> D | X-ray | Excitation energy, detector geometry, spatial resolution | Element mapping under matrix effect, self-absorption, pile-up | Estimate matrix composition, self-absorption correction factors | Correct matrix effects, dead time, pile-up |

---

### 3.13 Scientific Instrumentation (8 new modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 82 | `xray_crystallography` | X-ray Crystallography | F --> S --> D | X-ray | Wavelength, rotation range, detector distance, mosaicity | Structure factor extraction under absorption, radiation damage | Estimate unit cell, space group, absorption correction | Correct absorption, scaling, radiation damage compensation |
| 83 | `saxs` | Small-Angle X-ray Scattering | R --> D | X-ray | Beam size, q-range, sample-detector distance | Desmearing / model fitting under beam divergence, parasitic scatter | Estimate beam profile, background scatter, transmission | Correct beam smearing, background subtraction |
| 84 | `maldi_msi` | MALDI Mass Spec Imaging | S --> D | Ion | Laser spot size, step size, matrix application | Ion image reconstruction under matrix crystal inhomogeneity, mass drift | Estimate mass calibration drift, ion suppression map | Correct mass calibration, normalize ion suppression |
| 85 | `atom_probe` | Atom Probe Tomography | S --> D | Ion | Voltage/laser pulse, detection efficiency, FOV | 3D reconstruction under trajectory aberration, local magnification | Estimate tip shape evolution, local magnification | Correct reconstruction geometry, compositional bias |
| 86 | `cryo_em` | Cryo-EM Single Particle | C --> D | Electron | Voltage, defocus range, dose, ice thickness | CTF correction + 3D refinement under beam tilt, astigmatism, ice contamination | Estimate CTF per micrograph, beam tilt, ice thickness | Correct CTF, beam tilt, Ewald sphere curvature |
| 87 | `neutron_tomo` | Neutron Tomography | Pi --> D | Neutron | Beam flux, collimation ratio, rotation steps | FBP under beam hardening, scattering, gamma contamination | Estimate beam spectrum, scattering correction factor | Correct beam hardening, scatter, gamma filter |
| 88 | `proton_radiography` | Proton Radiography | Pi --> D | Proton | Beam energy, detector stack, angular acceptance | Most Likely Path reconstruction under MCS model error | Estimate scattering model parameters, energy loss | Correct MCS model, energy calibration |
| 89 | `muon_tomo` | Muon Tomography | Pi --> D | Muon | Detector layers, angular resolution, integration time | POCA / MLP reconstruction under angular resolution limit, noise | Estimate detector alignment, angular uncertainty | Correct detector alignment, improve track fitting |

---

### 3.14 Broader Experimental Science (8 new modalities)

| # | Modality ID | Full Name | Canonical DAG | Carrier | B1 Design | B2 Forward/Recon | B3 Identification | B4 Correction |
|---|-------------|-----------|---------------|---------|-----------|-----------------|-------------------|---------------|
| 90 | `adaptive_optics` | Adaptive Optics (Astronomy) | M --> C --> D | Photon | Wavefront sensor, DM actuator count, guide star, control bandwidth | PSF reconstruction under residual wavefront error, anisoplanatism | Estimate residual wavefront, Cn2 profile, wind speed | Correct AO loop parameters, update turbulence model |
| 91 | `seismic_tomo` | Seismic Tomography | P --> D | Acoustic/Seismic | Station array, frequency band, ray geometry | Travel-time / full-waveform inversion under velocity model error | Estimate velocity structure, source locations | Correct velocity model, relocate sources |
| 92 | `gravitational_wave` | Gravitational Wave Detection | P --> Sigma --> D | Gravitational | Arm length, laser power, mirror quality, seismic isolation | Matched filter under noise non-stationarity, glitch contamination | Estimate noise PSD, glitch morphology, calibration response | Correct calibration, glitch subtraction, noise model |
| 93 | `particle_calorimetry` | Particle Physics Calorimetry | R --> Sigma --> D | Particle | Absorber/scintillator layers, granularity, energy range | Energy/position reconstruction under inter-calibration error, non-linearity | Estimate cell-by-cell calibration, non-linearity curve | Correct inter-calibration, non-linearity compensation |
| 94 | `radio_astronomy` | Radio Aperture Synthesis | F --> S --> D | RF | Antenna configuration, bandwidth, polarization, uv-coverage | CLEAN / MEM imaging under baseline phase error, RFI | Estimate antenna gains, phase offsets, RFI contamination | Correct antenna calibration, RFI excision |
| 95 | `acoustic_emission` | Acoustic Emission Testing | P --> S --> D | Acoustic | Sensor placement, frequency range, threshold, timing | Source localization under velocity anisotropy, sensor coupling | Estimate wave velocity, sensor coupling quality, source type | Correct velocity model, improve localization |
| 96 | `magnetic_particle` | Magnetic Particle Imaging | M --> F --> D | Magnetic | FFP trajectory, drive field, selection field, harmonics | System function inversion under relaxation effects, background | Estimate system function, relaxation parameters, background | Correct system function, improve spatial resolution |
| 97 | `impedance_tomo` | Electrical Impedance Tomography | M --> D | Electric | Electrode count, current pattern, measurement protocol | Gauss-Newton / D-bar under contact impedance error, electrode position error | Estimate contact impedance, electrode positions, conductivity | Correct contact impedance, electrode model |

---

### 3.15 Particle Imaging (from existing registry)

Covered above: `neutron_tomo` (#87), `proton_radiography` (#88), `muon_tomo` (#89).

---

## 4. Summary: Modality Count by Category

| Category | Existing Registry | Expanded | Total | Medical Physicist Relevance |
|----------|:-:|:-:|:-:|---|
| Microscopy | 13 | 3 | 16 | Diagnostic imaging (pathology), nuclear medicine (autoradiography) |
| Compressive Imaging | 4 | 0 | 4 | Spectral tissue imaging, dose-efficient acquisition |
| Medical Imaging | 17 | 3 | 20 | **All 4 roles directly** -- see Section 5 |
| Coherent Imaging | 3 | 0 | 3 | Phase-contrast imaging, crystallography |
| Computational Photography | 2 | 0 | 2 | Endoscopic imaging, fundoscopy |
| Computational Optics | 2 | 0 | 2 | Light-field microscopy for pathology |
| Neural Rendering | 2 | 0 | 2 | Surgical planning, 3D anatomy |
| Electron Microscopy | 8 | 0 | 8 | Materials for devices, radiobiology |
| Clinical Optics | 3 | 0 | 3 | **Diagnostic imaging** -- ophthalmology |
| Depth Imaging | 3 | 0 | 3 | Surface-guided radiation therapy |
| Remote Sensing | 2 | 6 | 8 | Environmental monitoring, dose mapping |
| Industrial Inspection | 0 | 8 | 8 | QA of medical devices, accelerator components |
| Scientific Instrumentation | 3 | 5 | 8 | Proton/neutron therapy verification |
| Broader Experimental Science | 0 | 8 | 8 | MPI (tracers), EIT (lung monitoring) |
| **Total** | **62** | **33** | **95** | |

---

## 5. Medical Physicist Applications: How the 4 Benchmarks Serve 4 Professional Roles

Medical physicists are certified professionals in four subspecialties (per AAPM/CAMPEP/IPEM). Each role is served by all four benchmarks, but the emphasis differs.

### 5.1 Therapeutic Medical Physicist (Radiation Therapy)

**Primary modalities**: CT, CBCT, MRI (for planning), PET (for biological targeting), portal imaging (EPID), proton therapy imaging, brachytherapy imaging.

| Benchmark | Application in Therapeutic Physics |
|-----------|-----------------------------------|
| **B1** (Prompt-->Spec) | Design imaging protocol for treatment planning: "4D-CT for lung SBRT with 0.5mm resolution, respiratory gating" --> Spec defines scan parameters, gating strategy, dose limits. Design QA phantom imaging protocols for commissioning. |
| **B2** (Spec-->Recon) | Simulate the imaging chain from Spec to verify: does the designed CT protocol produce sufficient contrast for tumor delineation? Does the CBCT protocol provide adequate soft-tissue contrast for IGRT? Predict image quality before irradiating the patient. Generate feedback: "increase mA to reduce noise in mediastinal window" or "add iterative reconstruction to maintain quality at reduced dose." |
| **B3** (Data-->Spec) | Reverse-engineer imaging system parameters from QA phantom scans: given a CBCT dataset of a Catphan phantom, identify the actual geometric parameters (source-detector distance, gantry flex, detector offset) vs. nominal values. True-Spec = manufacturer calibration report. Essential for annual QA and post-service verification. |
| **B4** (Data+Spec-->Correction) | Correct CBCT geometric errors for accurate dose calculation: given daily CBCT images and the nominal Spec, correct for gantry sag, detector flex, and ring artifacts. Reconstruct artifact-free images for adaptive replanning. Feedback: "detector offset has drifted 0.3mm since last calibration -- recommend recalibration." |

**Key metrics**: Geometric accuracy (sub-mm for SRS/SBRT), HU accuracy (within 20 HU for dose calculation), dose efficiency (image quality per unit patient dose).

### 5.2 Diagnostic Imaging Medical Physicist

**Primary modalities**: CT, MRI, X-ray radiography, fluoroscopy, mammography, ultrasound, DEXA, angiography, OCT, fundus, endoscopy.

| Benchmark | Application in Diagnostic Imaging Physics |
|-----------|------------------------------------------|
| **B1** (Prompt-->Spec) | Design imaging protocols optimized for diagnostic task: "low-dose chest CT for lung cancer screening, CTDI_vol < 3 mGy" --> Spec defines kV, mA, pitch, reconstruction kernel, iterative reconstruction parameters. Design QA programs: specify phantom tests, acceptance criteria, testing frequency for each modality. |
| **B2** (Spec-->Recon) | Predict diagnostic image quality from protocol Spec before clinical use. Simulate: if we reduce dose by 30%, what happens to low-contrast detectability? If we switch from FBP to model-based iterative reconstruction, how does spatial resolution change? Generate feedback: "current protocol exceeds diagnostic reference level -- reduce mAs by 15% with no loss in CNR." |
| **B3** (Data-->Spec) | Determine actual system parameters from ACR phantom images or clinical images: given an MRI dataset, estimate actual coil sensitivity profiles, gradient nonlinearity, and geometric distortion. Compare against True-Spec (manufacturer calibration). Essential for accreditation (ACR, MQSA for mammography). |
| **B4** (Data+Spec-->Correction) | Correct for system imperfections in clinical images: given MRI data with known sequence parameters (Spec), correct for geometric distortion (for stereotactic procedures), intensity non-uniformity (for quantitative analysis), and ghosting. Feedback: "coil element #3 shows 12% sensitivity drop -- recommend coil service." |

**Key metrics**: Contrast-to-noise ratio (CNR), modulation transfer function (MTF), noise power spectrum (NPS), low-contrast detectability, geometric accuracy, patient dose metrics (CTDI, DLP, DAP, MGD).

### 5.3 Nuclear Medicine Medical Physicist

**Primary modalities**: PET, SPECT, gamma camera planar, PET/CT, PET/MR, SPECT/CT, radionuclide therapy dosimetry, molecular imaging.

| Benchmark | Application in Nuclear Medicine Physics |
|-----------|----------------------------------------|
| **B1** (Prompt-->Spec) | Design acquisition and reconstruction protocols: "FDG-PET/CT for lymphoma staging, 3-minute beds, TOF-OSEM, 4mm FWHM target resolution" --> Spec defines scanner geometry, crystal ring parameters, TOF resolution, attenuation correction method, scatter model. Design radionuclide therapy dosimetry protocols: "Lu-177 SPECT/CT for PRRT dosimetry at 24h, 96h, 168h post-injection." |
| **B2** (Spec-->Recon) | Simulate PET/SPECT acquisition chain: given the scanner Spec, predict noise-equivalent count rate (NECR), scatter fraction, and reconstructed spatial resolution. Test: if we reduce acquisition time by 50%, what happens to lesion detectability? Generate feedback: "current time-per-bed produces SUV coefficient of variation > 15% -- increase by 30s per bed position." |
| **B3** (Data-->Spec) | Determine system parameters from NEMA phantom data: given PET scanner data of NEMA IQ phantom, estimate actual spatial resolution, scatter fraction, sensitivity, count rate performance, and compare against True-Spec (manufacturer specifications). Essential for ACR accreditation, site qualification for clinical trials. |
| **B4** (Data+Spec-->Correction) | Correct PET/SPECT reconstructions for quantitative accuracy: given patient data and scanner Spec, correct for attenuation (CT-based mu-map errors), scatter (model-based correction errors), normalization (detector efficiency non-uniformity), randoms, dead time. Feedback: "detector block #47 normalization deviates > 5% from mean -- initiate PMT gain recalibration." For therapy dosimetry: correct SPECT quantification for accurate dose-volume histograms. |

**Key metrics**: Standardized uptake value (SUV) accuracy, spatial resolution (FWHM), scatter fraction, sensitivity, NECR, dead time correction accuracy, dose-volume histogram accuracy (for therapy).

### 5.4 Health Physics (Radiation Protection)

**Primary modalities**: Personnel dosimeters (TLD, OSL, film badge), area monitors, contamination survey instruments, environmental monitoring, portal monitors, whole-body counters.

| Benchmark | Application in Health Physics |
|-----------|-------------------------------|
| **B1** (Prompt-->Spec) | Design radiation monitoring systems: "personnel dosimetry program for interventional radiology suite, Hp(10) and Hp(0.07), monthly exchange" --> Spec defines dosimeter type, calibration source, algorithm, reporting threshold. Design shielding: "specify barrier requirements for PET/CT suite, 20 patients/day, controlled/uncontrolled area limits." |
| **B2** (Spec-->Recon) | Simulate dose distributions from Spec: given shielding design Spec, predict dose rates at all occupied locations. Simulate personnel dose accumulation under various workload scenarios. Generate feedback: "current shielding design produces 0.12 mSv/week in adjacent office -- exceeds design goal of 0.1 mSv/week, add 0.5 mm Pb to wall B." |
| **B3** (Data-->Spec) | Determine actual radiation environment from survey data: given area monitor readings and survey measurements, reconstruct the source distribution and shielding effectiveness. Compare against True-Spec (shielding design documents, source inventory). Essential for regulatory compliance surveys. |
| **B4** (Data+Spec-->Correction) | Correct dose estimates for actual conditions: given survey data and shielding Spec, correct for occupancy factors, workload changes, new equipment. Feedback: "measured dose rate 2x higher than predicted -- likely cause: missing backscatter from new equipment installation, recommend additional survey at 30 cm from wall C." For contamination: correct survey instrument readings for efficiency, background, and geometry factors. |

**Key metrics**: Dose equivalent rates (mSv/hr), annual dose limits compliance, contamination levels (Bq/cm2), shielding transmission factors, ALARA program effectiveness.

---

## 6. Analysis: Do the 4 Benchmarks Cover the 4-Level Targeting System?

### 6.1 Coverage Matrix

| PWM Component | B1 (Design) | B2 (Simulate) | B3 (Identify) | B4 (Correct) | Covered? |
|---------------|:-:|:-:|:-:|:-:|:-:|
| **Track 1: Correct** | | Scenario I-II | | **PRIMARY** | YES |
| **Track 2: Diagnose** | | Gate attribution | Gate identification | Feedback | YES |
| **Track 3: No-GT** | | | **PRIMARY** | When GT unavailable | YES |
| **Track 4: Design** | **PRIMARY** | Design validation | | | YES |
| **4-Scenario Protocol** | Design scenarios | Run Sc. I, II | Inform Sc. III | Run Sc. III, IV | YES |
| **Triad Law Gate 1** (Sampling) | Design sampling | Simulate sampling limits | Identify sampling | Diagnose sampling | YES |
| **Triad Law Gate 2** (Noise) | Design noise budget | Simulate noise impact | Identify noise level | Diagnose noise | YES |
| **Triad Law Gate 3** (Mismatch) | Design tolerance envelope | Simulate mismatch impact | Identify mismatch params | **Correct mismatch** | YES |
| **Red Team injection** | Novel requirements | Novel mismatch types | Unknown system configs | Compound failures | YES |
| **Anti-Goodhart scoring** | Pareto optimality | Prospective fidelity | Parameter accuracy | Recovery ratio rho | YES |
| **Compute efficiency (RoIC)** | Design for calibratability | Simulation cost | Identification cost | dB/GPU-hr | YES |
| **Uncertainty quantification** | Robustness margin | Noise propagation | Parameter CIs | Calibrated CIs | YES |

### 6.2 The Closed Loop

The four benchmarks form a **closed loop** that mirrors the complete lifecycle of an imaging system:

```
                    B1: Design
                   /          \
                  /            \
    B4: Correct  <              > B2: Simulate
                  \            /
                   \          /
                    B3: Identify
```

1. **B1 (Design)** produces a Spec from requirements
2. **B2 (Simulate)** validates the Spec by running it forward and reconstructing
3. **B3 (Identify)** reverse-engineers the Spec from real data (closing the sim-to-real gap)
4. **B4 (Correct)** fixes the Spec when reality deviates, reconstructs, and provides feedback that feeds back into B1 for the next design iteration

This loop implements the **Design-Make-Test-Iterate** cycle that the SolveEverything framework requires for industrialization.

### 6.3 What the 4 Benchmarks Add Beyond the Existing Targeting System

The existing PWM targeting system (LIP Arena) focuses primarily on **B4** (correction + reconstruction) and partially on **B2** (forward simulation for scenario generation). The 4-benchmark framework adds:

| Gap in Existing System | Addressed by Benchmark |
|------------------------|----------------------|
| No explicit design evaluation | **B1**: Spec generation from prompt, evaluated for feasibility and Pareto optimality |
| No system identification benchmark | **B3**: Reverse-engineering the Spec from data, evaluated against True-Spec |
| Spec is assumed, not generated | **B1** and **B3**: Spec is an output, not just an input |
| True-Spec concept not formalized | **B3** and **B4**: True-Spec as hidden ground truth for judging Spec quality |
| Feedback quality not evaluated | **B2** and **B4**: Feedback reports judged for actionability |
| No prompt-based interface | **B1** and **B3**: Natural language prompts as first-class inputs |

### 6.4 Verdict

**Yes, the four benchmarks can fully implement the 4-level targeting system structure.** Specifically:

1. **B1 covers Track 4 (Design)** -- the only benchmark that tests the ability to create an imaging system from scratch.

2. **B2 covers the forward-model validation layer** that underpins all tracks -- without faithful forward simulation (Scenario I), no other evaluation is meaningful. B2 also produces the diagnostic feedback that feeds Track 2.

3. **B3 covers Track 3 (No-GT) and the system identification aspect of Track 2** -- determining what system produced the data is a prerequisite for correcting it. The True-Spec mechanism makes this evaluation rigorous.

4. **B4 covers Track 1 (Correct) as the primary benchmark** and contributes to Tracks 2 and 3 through its feedback and no-GT correction capabilities.

The four benchmarks are not merely sufficient -- they are **necessary and complementary**. Removing any one leaves a gap:
- Without B1: no design evaluation; the targeting system cannot test whether ISA can create systems, only fix them.
- Without B2: no forward-model validation; the chain from Spec to measurement is untested.
- Without B3: no system identification; the chain from measurement to Spec is untested.
- Without B4: no correction evaluation; the core ISA capability (autonomous mismatch correction) is untested.

Together, they close the loop and provide the complete infrastructure for PWM's targeting system across all 95+ imaging modalities, all 4 medical physicist roles, and all maturation levels from L0 to L5.

---

## 7. Implementation: True-Spec Schema for Each Benchmark

### 7.1 True-Spec Structure

Every benchmark dataset should ship with a True-Spec file that enables objective evaluation:

```yaml
# true_spec.yaml -- hidden from contestants, used by LIP Arena for scoring
version: "1.0"
modality: cassi
benchmark: B4  # which benchmark this True-Spec serves

# Exact system parameters (hidden ground truth)
system:
  primitives:
    - {type: M, id: coded_mask, params: {density: 0.5, shift_x: 1.47, shift_y: -0.23, rotation_deg: 0.31}}
    - {type: W, id: spectral_dispersion, params: {slope: 2.01, offset: 0.04}}
    - {type: Sigma, id: spectral_sum, params: {}}
    - {type: D, id: cmos_detector, params: {gain: 1.02, read_noise: 5.1, dark_current: 0.3}}
  noise:
    type: poisson_gaussian
    params: {shot: true, read_sigma: 5.1, dark_rate: 0.3}

# Mismatch tolerance envelope (disclosed to contestants)
mismatch_ranges:
  shift_x: {min: -3.0, max: 3.0}
  shift_y: {min: -3.0, max: 3.0}
  rotation_deg: {min: -2.0, max: 2.0}
  dispersion_slope: {min: 1.5, max: 2.5}
  gain: {min: 0.9, max: 1.1}

# Improvement recommendations (for judging B4 feedback quality)
recommendations:
  - {priority: 1, action: "recalibrate mask alignment", expected_gain_dB: 5.0}
  - {priority: 2, action: "re-measure dispersion curve with calibration lamp", expected_gain_dB: 1.2}
  - {priority: 3, action: "reduce read noise via cooling to -20C", expected_gain_dB: 0.3}

# Evaluation weights
scoring:
  parameter_rmse_weight: 0.30
  reconstruction_quality_weight: 0.30
  feedback_quality_weight: 0.20
  compute_efficiency_weight: 0.10
  uncertainty_calibration_weight: 0.10
```

### 7.2 Benchmark Dataset Package Structure

```
benchmark_pack/
  modality.yaml          # Modality metadata (DAG, carrier, category)
  B1/
    prompt.txt           # Natural language prompt
    original_spec.yaml   # Optional previous spec (may be null)
    true_spec.yaml       # Hidden: ground truth for what the spec should look like
  B2/
    spec.yaml            # Complete spec to simulate from
    true_spec.yaml       # Hidden: exact noise/mismatch parameters injected
  B3/
    dataset/
      y.npy              # Measurement data
      H_nominal.npz      # Nominal system matrix (if applicable)
      metadata.json       # Dataset metadata
    prompt.txt           # Natural language description
    original_spec.yaml   # Optional previous spec
    true_spec.yaml       # Hidden: exact system parameters
  B4/
    dataset/
      y.npy
      H_nominal.npz
      metadata.json
    spec.yaml            # Imperfect spec (with mismatch errors)
    true_spec.yaml       # Hidden: exact parameters + recommendations
```

---

## 8. Cross-Reference: Modality Categories and Medical Physicist Roles

| Category | Therapeutic MP | Diagnostic Imaging MP | Nuclear Medicine MP | Health Physics |
|----------|:-:|:-:|:-:|:-:|
| Microscopy | Radiobiology | Pathology imaging QA | Autoradiography | Bioassay imaging |
| Compressive Imaging | Dose-efficient therapy imaging | Spectral tissue imaging | Compressed PET/SPECT | |
| Medical Imaging (X-ray) | **CBCT, portal imaging** | **CT, radiography, mammo, fluoro** | PET/CT attenuation | **Shielding verification** |
| Medical Imaging (MRI) | **MR-guided RT planning** | **Clinical MRI QA** | PET/MR | RF safety |
| Medical Imaging (Nuclear) | **PET for biological targeting** | | **PET, SPECT, therapy dosimetry** | Contamination surveys |
| Medical Imaging (Ultrasound) | **US-guided brachytherapy** | **Clinical US QA** | | |
| Clinical Optics | | **Fundus, OCT, OCTA** | | |
| Electron Microscopy | | | | Dosimeter characterization |
| Depth Imaging | **Surface-guided RT (SGRT)** | | | |
| Remote Sensing | | | Environmental monitoring | **Area monitoring** |
| Industrial Inspection | Accelerator component QA | Medical device QA | Source QC | **Shielding survey** |
| Scientific Instrumentation | Proton/neutron therapy verify | | | |
| Broader Experimental Science | MPI-guided therapy | EIT (ventilation) | | |

---

## 9. References

1. PWM Flagship Paper -- [Typed Primitives and OperatorGraph IR](https://github.com/integritynoble/Physics_World_Model/blob/master/papers/pwm_flagship/main.pdf)
2. PWM Targeting System -- [`docs/targeting_system.md`](targeting_system.md) (LIP Arena specification)
3. PWM Imaging Modality Registry -- [`docs/imaging_modalities.md`](imaging_modalities.md) (64-modality registry)
4. PWM Operator Mode -- [`docs/operator_mode.md`](operator_mode.md) (calibration pipeline)
5. PWM Canonical Primitives -- [`docs/plan_canonical_primitives.md`](plan_canonical_primitives.md) (10-primitive basis)
6. PWM Purpose (ISA) -- [`docs/purpose.md`](purpose.md) (Industrial Intelligence Stack)
7. SolveEverything Framework -- [https://solveeverything.org/](https://solveeverything.org/)
8. AAPM Task Groups -- TG-142 (linac QA), TG-66 (image QA), TG-174 (CT QA)
9. CAMPEP Standards -- Medical physicist subspecialty definitions
