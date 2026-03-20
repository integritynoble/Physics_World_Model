# Lensless Imaging Prototype Design
## For Hardware Validation of the FPB Design Framework

### Purpose
Build a physical lensless imaging system designed entirely from the FPB framework's
`spec.md` specification, then compare real reconstruction quality against simulation
predictions. This demonstrates that the framework guides real instrument construction,
not just simulation (Nature reviewer point #8).

---

## 1. System Architecture

We propose a **three-stage prototype** that progressively validates more complex
FPB primitive chains:

### Stage A: 2D Lensless ($C \to D$) — Baseline
Single-shot 2D imaging through a random phase mask (diffuser).
- **FPB chain:** Convolve(C, psf=phase_mask) → Detect(D)
- **Compression:** 1:1 (no compression)
- **Simulated PSNR:** 43.7 dB (ADMM+TV)
- **Purpose:** Validate basic framework on simplest possible system

### Stage B: 3D Lensless ($\Phi_z \to \Sigma \to D$) — Novel Design
Single-shot 3D depth recovery via defocus-modulated diffuser.
- **FPB chain:** Convolve_z(C, psf=diffuser(z)) → Sum(Σ) → Detect(D)
- **Compression:** 8:1 (8 depth planes from 1 image)
- **Simulated PSNR:** 20.3 dB (R-L)
- **Purpose:** Demonstrate novel system design capability

### Stage C: Spectral Lensless ($M \to W \to C \to \Sigma \to D$) — Full Novel Chain
Single-shot spectral imaging with coded mask + dispersive element + diffuser.
- **FPB chain:** Modulate(M) → Disperse(W) → Convolve(C) → Sum(Σ) → Detect(D)
- **Compression:** 8:1 (8 spectral bands)
- **Simulated PSNR:** 36.3 dB (FISTA+TV)
- **Purpose:** Validate longest novel chain on hardware

---

## 2. spec.md Specifications

### Stage A: 2D Lensless
```yaml
modality: lensless_imaging
carrier: photon
geometry: single_shot, 512x512
object: 512x512 2D image (natural scenes)
forward_model: Convolve(C, psf=phase_mask) -> Detect(D)
noise: Poisson I_0=10000 + Gaussian sigma=5 (read noise)
target: PSNR >= 30 dB (real-world target, lower than simulation)
system_elements:
  source: broadband LED (white, 400-700nm)
  optics: random phase diffuser (220-grit ground glass, ~50μm feature scale)
  detector: Sony IMX477 (Raspberry Pi HQ camera, 4056×3040, 1.55μm pixel)
  working_distance: 5 cm sensor-to-diffuser, 30 cm diffuser-to-scene
```

### Stage B: 3D Lensless
```yaml
modality: 3d_lensless
carrier: photon
geometry: single_shot, 512x512, n_depths=8
object: 512x512x8 3D volume (depth-separated objects)
forward_model: Convolve_z(C, psf=diffuser(z)) -> Sum(Sigma) -> Detect(D)
noise: Poisson I_0=10000 + Gaussian sigma=5
target: PSNR >= 15 dB
system_elements:
  source: broadband LED (white)
  optics: random phase diffuser (220-grit ground glass)
  detector: Sony IMX477 (4056×3040, 1.55μm pixel)
  depth_range: 20-60 cm (8 planes at 5 cm intervals)
  calibration: PSF measured at each depth plane via point source scan
```

### Stage C: Spectral Lensless
```yaml
modality: spectral_lensless
carrier: photon
geometry: single_shot, 512x512, n_bands=8
object: 512x512x8 spectral datacube
forward_model: Modulate(M, mask=binary_random) -> Disperse(W, prism) ->
               Convolve(C, psf=diffuser) -> Sum(Sigma) -> Detect(D)
noise: Poisson I_0=8000 + Gaussian sigma=5
target: PSNR >= 25 dB
system_elements:
  source: broadband halogen lamp (continuous spectrum 400-700nm)
  optics:
    - coded_mask: DMD (Texas Instruments DLP3000, 608×684 micromirrors)
      OR printed binary mask on transparency film (low-cost option)
    - dispersive_element: N-BK7 prism (30° apex, ~40nm/mm dispersion)
      OR diffraction grating (300 lines/mm transmission)
    - diffuser: 220-grit ground glass
  detector: Sony IMX477 (4056×3040, 1.55μm pixel)
```

---

## 3. Bill of Materials (BOM)

### Core Components (Stage A, ~$150)
| Component | Part | Est. Cost |
|-----------|------|-----------|
| Sensor | Raspberry Pi HQ Camera + Pi 4B | $75 |
| Diffuser | 220-grit ground glass, 25mm dia | $15 |
| LED source | White LED array (5W) | $10 |
| Optical mount | Thorlabs cage system (30mm) | $40 |
| Calibration target | USAF 1951 resolution target | $10 |

### Additional for Stage B (+$50)
| Component | Part | Est. Cost |
|-----------|------|-----------|
| Translation stage | Manual z-stage for PSF calibration | $35 |
| Point source | Fiber-coupled LED + pinhole (50μm) | $15 |

### Additional for Stage C (+$200-500)
| Component | Part | Est. Cost |
|-----------|------|-----------|
| Coded mask (low-cost) | Printed binary pattern on film | $5 |
| Coded mask (high-end) | DMD evaluation module (DLP3000) | $350 |
| Dispersive element | N-BK7 prism, 25mm | $40 |
| Narrowband filters | 8× bandpass (for calibration) | $160 |

**Total: $200 (Stage A+B low-cost) to $700 (full Stage C with DMD)**

---

## 4. Calibration Protocol

### PSF Measurement (Stages A & B)
1. Place point source (pinhole + LED) at known distance z
2. Capture raw sensor image (this IS the PSF at distance z)
3. Repeat for each depth plane (Stage B: 8 depths)
4. Store PSFs in calibration file

### Spectral Calibration (Stage C)
1. Use narrowband filters (8 bands: 425, 450, 500, 525, 550, 575, 625, 675 nm)
2. For each band: illuminate uniform scene, capture coded+dispersed measurement
3. This gives the spectral response matrix
4. Measure dispersion shift per wavelength

### Adjoint Validation
For each calibrated forward model A:
1. Generate random x, compute y = Ax
2. Generate random y', compute x' = A^T y'
3. Verify <y, y'> ≈ <Ax, y'> = <x, A^T y'> (inner product test)
4. Relative error must be < 1e-6

---

## 5. Experimental Protocol

### Phase 1: Simulated-to-Real Comparison
For each stage (A, B, C):
1. Calibrate PSF on real hardware
2. Capture N=20 test scenes (natural objects, resolution targets, etc.)
3. Reconstruct using ADMM+TV, FISTA+TV, R-L (same algorithms as simulation)
4. Compare real PSNR against simulation PSNR
5. Compute each ε term from Theorem 1:
   - ε_FPB: compare real PSF vs ideal model PSF
   - ε_param: measure via calibration perturbation
   - ε_unmod: compare Tier-1 vs Tier-3 models
6. Verify: real MSE ≤ predicted MSE bound (within τ factor)

### Phase 2: Novel Design Validation (Stage B)
1. Place 3-5 objects at different depths (playing cards, USAF targets)
2. Capture single 2D image
3. Reconstruct 3D volume
4. Verify depth separation and PSNR at each plane
5. This is THE key result: a system designed by the framework, built, and working

### Phase 3: Specification-Dominance Verification
1. Run all 5 reconstruction algorithms on real data
2. Compute inter-method CoV on real data
3. Compare against simulation CoV (should be <6% for 2D lensless)
4. This validates the central finding on real hardware

---

## 6. Why Each Collaborator is Ideal

### Laura Waller (UC Berkeley)
- **Invented DiffuserCam** — the exact system in Stage A
- Has existing hardware, 100K-image datasets, and reconstruction codebase
- **Role:** Stage A & B validation, real PSF characterization
- **What we bring:** Formal error decomposition framework she hasn't published
- **Paper contribution:** "Real-data lensless validation" section, co-author

### Liang Gao (UCLA)
- Expert in **snapshot compressive imaging**, spectral + temporal coding
- Published extensively on CASSI-type and coded aperture systems
- **Role:** Stage C (spectral lensless) — has DMD and prism setups
- **What we bring:** Novel spectral lensless chain ($M \to W \to C \to \Sigma \to D$)
  that his group hasn't explored
- **Paper contribution:** Spectral lensless prototype, co-author

### David Brady (University of Arizona)
- Pioneer of **coded aperture imaging** and compressive imaging theory
- Built some of the first CASSI and gigapixel camera systems
- **Role:** Senior validation advisor, independent verification
- **What we bring:** Formalization of design principles his group uses intuitively
- **Paper contribution:** Independent verification letter or co-authorship

### Alternative Collaborators
- **Nick Antipa** (formerly Waller Lab, now UCSD) — DiffuserCam co-inventor
- **Ashok Veeraraghavan** (Rice) — FlatCam / lensless expert
- **Xin Yuan** (Westlake) — CACTI / snapshot compressive imaging
- **Gordon Wetzstein** (Stanford) — neural holography, light field displays

---

## 7. Collaboration Pitch (Draft Email)

Subject: Collaboration: Formal Design Framework Validated on DiffuserCam Hardware

Dear Prof. Waller,

I am writing about a potential collaboration to validate a formal imaging system
design framework on real lensless imaging hardware. My recent work (preprint at
[link]) introduces a representation theorem showing that 11 canonical primitives
suffice to describe any imaging forward model, and a design-to-real error
decomposition (5 independent terms) that provides bounded reconstruction
guarantees.

The key finding is that forward-model specification — not algorithm choice —
is the primary quality determinant for well-conditioned systems (inter-algorithm
PSNR CoV < 6%). Your DiffuserCam system is the ideal testbed because:

1. It maps cleanly to a 2-primitive chain (C → D)
2. You already have calibrated PSFs and 25K+ paired measurements
3. The 3D extension (Φ_z → Σ → D) demonstrates a novel design from the framework

I propose a two-phase experiment: (a) validate the error decomposition on existing
DiffuserCam data, and (b) build a 3D lensless prototype guided entirely by the
framework's spec.md specification. The goal is a Nature-level demonstration that
formal specification can replace expert intuition in imaging system design.

The framework code is open-source. I would value your hardware expertise and
experimental validation — the simulation results predict 43.7 dB for 2D lensless
and 20.3 dB for 3D (8:1 compression).

Best regards,
Chengshuai Yang

---

## 8. Expected Paper Impact

### Before prototype:
- 6 real-data modalities, all established
- Novel designs validated only in simulation
- Reviewer concern: "framework produces simulations, not instruments"

### After prototype (Stages A+B):
- 8+ real-data modalities (adding lensless + upgraded ultrasound/OCT)
- 1 novel design (3D lensless) validated on real hardware
- Error decomposition verified in practice (each ε term measured)
- Specification-dominance finding confirmed on real data
- Direct before/after: spec.md → hardware → measured PSNR vs predicted PSNR

### Nature narrative:
"We designed a 3D lensless camera from a one-sentence description using the
framework, built it for $200, and achieved [X] dB PSNR — within [τ]× of the
theoretical bound. The 5-term error decomposition correctly predicted the dominant
error source (ε_param from PSF calibration) and the corrective action
(depth-resolved calibration) that improved PSNR by [Y] dB."
