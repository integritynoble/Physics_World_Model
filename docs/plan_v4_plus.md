# PLAN v4+ — Remaining Imaging Modalities (beyond PLAN_v4)

This plan covers *additional* imaging modalities **not already scheduled** in `PLAN_v4.md`.

## 0) Scope: what is excluded (already in PLAN_v4)

PLAN_v4 already schedules (Tier A/B/C) the following modalities, so **do not include them here**:
- Tier A: SPC, CACTI, CASSI, CT, MRI, Ultrasound, SEM, TEM
- Tier B: PET, SPECT, Electron Tomography, STEM
- Tier C (smoke only): Widefield, Confocal, SIM, Lensless, Light-sheet, Ptychography, Holography, NeRF, 3DGS, Matrix, Panorama, Light Field, Integral, Phase Retrieval, FLIM, Photoacoustic, OCT, FPM, DOT, Widefield Low-Dose

(Reference list appears in `PLAN_v4.md` under "P5.3: Staged acceptance gate".)

---

## 1) Success points (what "works well" means)

For **every new modality** added under this plan, the following must be true:

### S1. Graph correctness
- Compiles to canonical chain: **Source -> ElementNodes -> Sensor -> Noise -> y**.
- Ports / TensorSpecs validate: shape, dtype, units, coordinate frame.

### S2. End-to-end modes
- **Mode S (simulate)**: given (x, theta, psi, phi) produce y.
- **Mode I (invert)**: given y recover x_hat with a baseline solver.
- **Mode C (calibrate)**: at least **one** mismatch parameter is identifiable and improves NLL (or proxy objective).

### S3. Operator-correction ready
- "Measured y + imperfect A0" path works: build `CorrectedOperator(A0, correction_nodes, theta)` and show objective improves after fitting.

### S4. CasePack + acceptance
- A **synthetic CasePack generator** exists (minimal but deterministic).
- Acceptance tests exist:
  - Forward sanity (finite, nonneg where appropriate)
  - Baseline recon threshold (per-modality)
  - Calibration improvement (NLL or proxy)
  - Reproducibility (seed + hash)

### S5. Registries + prompt UX
- Modality registered (YAML) with:
  - default template
  - default solver
  - default noise model
  - mismatch parameter list + priors/bounds
  - keywords for prompt routing

---

## 2) Universal onboarding workflow (Claude should implement as a reusable pipeline)

**For each new modality:**
1. **Define StateSpec** (what travels): field/flux/phase-space/k-space/event list.
2. **Write SourceSpec + ExposureBudget** defaults.
3. **Compose minimal physics chain** (Tier 0/1): 2-4 element nodes.
4. **Add Sensor + Noise**:
   - photon/electron counting -> Poisson or Poisson+Gaussian
   - RF complex samples -> complex Gaussian
   - ToF events -> Poisson point process (start as binned Poisson)
5. **Add 1-3 mismatch knobs** with bounds + priors + drift model.
6. **Pick baseline inversion** (FBP/FFT/beamform/least-squares/TV-FISTA).
7. **Add CasePack generator** + acceptance tests.
8. **Add prompt keywords** + docs snippet.

This makes scaling to "many modalities" a mostly data/registry task.

---

## 3) Engineering primitives needed for "new families" (implement once, reuse everywhere)

### 3.1 State extensions (core/ir)
Add (or standardize) these state variants:
- `WaveFieldState` (complex field + wavelength/frequency axes)
- `RayBundleState` (rays: origin, direction, weight)
- `EventListState` (time-stamped photon/electron events; can be binned initially)
- `KSpaceState` (complex k-space samples + trajectory)
- `PhaseSpaceState` (position, angle, energy) for charged/neutral particles

### 3.2 New element node families (graph/primitives.py)
Implement primitives that unlock lots of modalities:
- **ScanTrajectory**: raster/spiral/line scan + dwell/exposure mapping
- **TimeOfFlightGate**: ToF binning + timing jitter
- **CollimatorModel**: SPECT/pinhole/endoscopy-style acceptance cones (generic)
- **FluoroTemporalIntegrator**: motion blur + frame integration for dynamic systems
- **FluorescenceKinetics**: lifetime / blinking / saturation (simple ODE first)
- **PolarizationOptics**: Jones/Mueller (optional; can start as "tags")
- **DepthOptics**: thin-lens + distortion + rolling shutter (for cameras/LiDAR)
- **DiffractionCamera**: far-field diffraction |FFT|^2 with detector PSF

### 3.3 Sensor primitives
- **SPAD/ToF sensor**: photon -> timestamp histogram + dead time (tier0)
- **Energy-resolving detector**: EELS-like spectrum bins (tier0)
- **FiberBundleSensor**: endoscope bundle mapping (permutation + blur)

---

## 4) Modality roadmap (beyond PLAN_v4)

Below are high-impact modality families NOT covered by PLAN_v4, grouped to maximize code reuse.

### Phase R1 -- X-ray "variants" (big market, minimal new physics)
**Modalities:**
- Fluoroscopy (dynamic radiography)
- Mammography
- DEXA (dual-energy)
- Cone-beam CT (CBCT) *as a variant of CT but separate registry entry*
- Angiography (as fluoroscopy + iodine contrast model)

**New primitives:**
- `FluoroTemporalIntegrator`
- optional `DualEnergyBeerLambert` (DEXA) with two spectra

**Mismatch knobs:**
- geometry drift (source-detector pose), detector gain/offset drift, scatter scale, motion blur

**Solvers:**
- radiography: log-inversion + denoise
- CBCT: FDK baseline (or reuse FBP + cone geometry approximations)

**Tests:**
- synthetic phantom + known spectrum
- calibration: gain drift and geometry shift improves NLL

---

### Phase R2 -- Ultrasound "modes" (still acoustic, huge installed base)
(You already have B-mode planned in PLAN_v4; here we add additional modes.)
**Modalities:**
- Doppler ultrasound (color / spectral)
- Shear-wave elastography

**New primitives:**
- `DopplerEstimator` (autocorrelation / FFT on slow-time)
- `ElasticWaveModel` (tier0 proxy: wave speed map affects phase)

**Mismatch knobs:**
- speed-of-sound map, PRF/timing errors, probe element sensitivity

**Solvers:**
- Doppler: FFT-based velocity estimate
- Elastography: inversion from displacement phase (regularized LS)

---

### Phase R3 -- MRI "applications" (same core physics, new recon targets)
(PLAN_v4 includes "MRI enhanced"; here we add modality entries that reuse the same operator.)
**Modalities:**
- fMRI (BOLD time series)
- MR Spectroscopy (MRS)
- Diffusion MRI (DTI as parameter-map inverse problem)

**New primitives:**
- `SequenceBlock` registry: EPI, MRS readout, diffusion gradients (parameterized)
- `PhysiologyDrift` (low-rank temporal drift model)

**Mismatch knobs:**
- B0 drift, trajectory errors, motion, coil sensitivity drift

**Solvers:**
- fMRI: low-rank + sparse / temporal filtering
- MRS: spectral peak fitting
- DTI: tensor fitting

---

### Phase R4 -- Advanced optical microscopy (high publish value)
**Modalities:**
- Two-photon / three-photon microscopy
- STED microscopy (saturation depletion)
- Single-molecule localization (PALM/STORM)
- TIRF microscopy
- Polarization microscopy (optional)

**New primitives:**
- `NonlinearExcitation` (intensity^n)
- `SaturationDepletion` (STED)
- `BlinkingEmitterModel` (PALM/STORM events)
- `EvanescentFieldDecay` (TIRF)

**Mismatch knobs:**
- PSF width/aberrations, illumination pattern phase, saturation parameter drift, background

**Solvers:**
- deconv + PnP for multiphoton/TIRF
- localization (MLE) for PALM/STORM
- STED: deconvolution with effective PSF model

---

### Phase R5 -- Clinical optics / endoscopy (real adoption pathways)
**Modalities:**
- Endoscopy (fiber bundle imaging)
- Fundus camera / retinal imaging
- OCT angiography (OCTA) *(variant of OCT, separate registry key)*

**New primitives:**
- `FiberBundleSensor`
- `VesselFlowContrast` (OCTA: decorrelation proxy)
- `SpecularReflectionModel` (endoscopy)

**Mismatch knobs:**
- bundle permutation error, per-core gain, motion between frames, dispersion drift (OCTA inherits OCT)

**Solvers:**
- bundle demosaic + deblur
- OCTA: temporal decorrelation + denoise

---

### Phase R6 -- Depth / time-of-flight imaging (huge robotics market)
**Modalities:**
- ToF camera (phase or pulse)
- LiDAR (scanning)
- Structured-light depth camera

**New primitives:**
- `TimeOfFlightGate`
- `ScanTrajectory`
- `StructuredLightProjector` (pattern -> depth via triangulation proxy)

**Mismatch knobs:**
- timing offset/jitter, scan angle bias, rolling shutter skew, reflectance-dependent bias

**Solvers:**
- depth from ToF histogram (argmax / matched filter)
- LiDAR: point cloud reconstruction + denoise

---

### Phase R7 -- Radar / SAR & Sonar (optional but extends "wave" universality)
**Modalities:**
- Synthetic Aperture Radar (SAR)
- Sonar imaging (basic beamforming)

**New primitives:**
- `SARBackprojection` (tier0/1)
- reuse `BeamformDelay` for sonar

**Mismatch knobs:**
- platform trajectory error, timing drift, sound speed for sonar

---

### Phase R8 -- Electron "spectroscopy / diffraction" (research instruments)
(PLAN_v4 covers SEM/TEM/STEM/ET imaging. Here: extra electron modalities.)
**Modalities:**
- Electron diffraction / 4D-STEM (diffraction patterns per scan point)
- EBSD (electron backscatter diffraction)
- EELS (energy loss spectrum imaging)
- Electron holography (phase retrieval with biprism)

**New primitives:**
- `DiffractionCamera`
- `EnergyResolvingDetector`
- `ReciprocalSpaceGeometry` (EBSD pattern mapping)

**Mismatch knobs:**
- camera length, energy scale drift, scan distortion, detector MTF

**Solvers:**
- phase retrieval / pattern fitting (tier0)
- spectral unmixing for EELS

---

### Phase R9 -- Neutral/ion particle imaging (niche, but strong "PWM universality" demo)
**Modalities:**
- Neutron radiography / tomography
- Proton radiography (therapy QA / research)
- Muon tomography (security/materials)

**New primitives:**
- `ParticleAttenuation` (Beer-Lambert analogue with energy-dependent cross sections)
- `MultipleScatteringKernel` (tier0)
- `TrackDetectorSensor` (muon tracks)

**Mismatch knobs:**
- spectrum/energy calibration, detector efficiency, scattering kernel width

**Solvers:**
- FBP-like tomography (for neutron/proton)
- track-based reconstruction (for muon) -- start with LS

---

## 5) Acceptance tiers (recommended)

To keep the project controllable, add *another* staged gate for these new modalities:

- **Tier A2 (must be solid):** R1 + R6 (X-ray variants + ToF/LiDAR)
  *Reason:* biggest market + easiest to validate quickly.
- **Tier B2:** R2 + R5 (ultrasound modes + endoscopy/ophthalmic)
- **Tier C2:** R3 + R4 (MRI apps + advanced microscopy)
- **Tier D2:** R7-R9 (radar/sonar + extra electron + particle)

Each tier has the same S1-S5 criteria, but different baseline thresholds.

---

## 6) Concrete coding checklist for Claude (copy/paste)

1. **IR/State**
   - Add state classes: WaveFieldState, EventListState, PhaseSpaceState, KSpaceState extensions
   - Add TensorSpec/unit checks for new axes (time, energy, angle)

2. **Primitives**
   - Implement ScanTrajectory, TimeOfFlightGate, FluoroTemporalIntegrator, DualEnergyBeerLambert
   - Implement SPADToFSensor, FiberBundleSensor, EnergyResolvingDetector, DiffractionCamera

3. **Templates**
   - Add graph templates + modality registry entries for:
     - fluoroscopy, mammography, dexa, cbct, angiography
     - tof_camera, lidar, structured_light
     - endoscopy, fundus, octa
     - doppler_ultrasound, elastography
     - electron_diffraction, ebsd, eels, electron_holography
     - neutron_tomo, proton_radiography, muon_tomo

4. **Mismatch + calibration**
   - Add mismatch ParamSpecs (bounds + priors) per modality
   - Ensure Mode C improves NLL/proxy on at least 1 knob per modality

5. **Solvers**
   - Add baseline solver plugins:
     - log-inversion + denoise, FDK (CBCT), ToF matched filter, SAR backprojection
     - localization MLE (PALM/STORM), peak fitting (MRS), doppler FFT

6. **CasePacks + tests**
   - Add synthetic CasePack generators for each new modality
   - Add `tests/test_new_modalities_acceptance_tierA2.py` etc.

7. **Prompt UX**
   - Expand prompt keywords
   - Add "modality family defaults" (e.g., ToF -> ToF sensor + Poisson model)

---

## 7) Notes on keeping it tractable

- Treat "variants" (mammography/fluoro/DEXA/CBCT/OCTA) as **separate registry entries** but **shared operators**.
- Start every new family at Tier0/1 approximations, then add Tier2 plugins later.
- Don't add a modality unless it has **one** identifiable mismatch knob + CasePack test.
