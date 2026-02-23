# PWM Per-Modality Benchmark Specifications

> **Scope**: Deep, actionable benchmark specifications (B1-B4) for all **168 imaging modalities**
> across **19 categories**. Each modality entry includes: example prompts, mismatch parameter
> ranges, True-Spec fields, correction targets, expected recovery metrics, and improvement suggestions.
>
> **Companion doc**: [`pwm_medical_physicist_targets.md`](pwm_medical_physicist_targets.md) for architecture overview.

---

## 0. Advisory: Modality Expansion & Benchmark Improvement Strategy

### 0.1 Expansion Summary

**Previous state**: 97 modalities across 13 categories.
**Current state**: **168 modalities across 19 categories.**

| Change | Count |
|--------|------:|
| Existing modalities retained | 97 |
| Added to existing categories | 42 |
| New categories created | 6 |
| New modalities in new categories | 29 |
| **Grand total** | **168** |

**6 new categories**: Spectroscopy & Spectral Imaging (8), Ultrafast Imaging (4), Quantum Imaging (3), Multi-modal Fusion (6), Scanning Probe Microscopy (4), Astronomy & Space Imaging (4).

**Expansion rationale**: Every addition satisfies at least one of: (a) new DAG topology, (b) new mismatch physics, (c) large user community. The Finite Primitive Basis theorem's 10 canonical types still cover all 168 modalities — no new primitives needed.

### 0.2 How to Improve Each Benchmark

**8 strategies** — apply per modality:

| # | Strategy | Applies to | How |
|---|----------|------------|-----|
| 1 | **Real datasets over synthetic** | B2, B3, B4 | Replace Gaussian phantoms with published experimental data (Zenodo, TCIA, public repos) |
| 2 | **Compound mismatch** | B2, B4 | Test 3+ params simultaneously, not just single-param |
| 3 | **Adversarial mismatch (Red Team)** | B2, B3, B4 | Worst-case mismatch via optimization |
| 4 | **Cross-modality transfer** | B1 | Adapt Spec across related modalities via prompt engineering |
| 5 | **Time-varying mismatch** | B3, B4 | Parameters drift across frames/slices |
| 6 | **Uncertainty quantification** | B3, B4 | Require confidence intervals on estimated params |
| 7 | **Multi-scale evaluation** | B2, B4 | PSNR + SSIM + FSIM + domain-specific metrics |
| 8 | **Feedback actionability scoring** | B2, B4 | Score feedback for actionability, not just correctness |

### 0.3 Benchmark Maturity Levels

| Level | B1 | B2 | B3 | B4 |
|-------|----|----|----|----|
| **M0** (Template) | Prompt template | Forward model template | DAG template | Correction template |
| **M1** (Synthetic) | Prompt tested | Single-param mismatch | Synthetic True-Spec | Single-param correction |
| **M2** (Compound) | Multiple variants | Compound mismatch, 3+ params | Compound identification | Compound correction, rho measured |
| **M3** (Real data) | Grounded in real protocols | Real experimental data | Real True-Spec | Real data, rho >= 0.80 |
| **M4** (Adversarial) | Adversarial attacks | Red Team injection | Adversarial identification | Adversarial + live feedback |

---

## 1. Microscopy (24 modalities)

---

### 1.1 Widefield Fluorescence Microscopy (`widefield`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M3

#### B1: Design
**Example prompt**: "Design widefield fluorescence for GFP-labeled fixed cells: 60x oil, NA 1.4, emission 500-550 nm, pixel 100 nm, FOV 80 um, sCMOS."

| Design Parameter | Range | Unit |
|------------------|-------|------|
| Objective NA | 0.4 - 1.49 | - |
| PSF sigma (lateral) | 0.8 - 4.0 | px |
| Emission wavelength | 400 - 800 | nm |
| Read noise | 1.0 - 10.0 | e- |

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| PSF sigma | 2.0 | [1.2, 3.5] | px |
| Background level | 50 | [0, 200] | counts |
| Gain | 1.0 | [0.85, 1.15] | - |
| Flatfield non-uniformity | 0% | [0%, 15%] | peak-to-peak |
| Photobleaching rate | 0 | [0, 0.05] | per frame |

**Solvers**: Richardson-Lucy, PnP-HQS, Wiener. **Scenario I PSNR**: 30-38 dB. **Scenario II drop**: 1-5 dB.

#### B3: True-Spec
PSF sigma_x (2.13), sigma_y (2.07), background (47.3), read noise (5.8), gain (1.03).

#### B4: Correction
**Expected rho**: >= 0.85. **PSNR gain**: +1 to +5 dB.
**Improvement**: Add compound mismatch (PSF + background + flatfield), depth-dependent PSF.

---

### 1.2 Low-Dose Widefield (`widefield_lowdose`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M1

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Photon rate alpha | 100 | [10, 500] | photons/px |
| Read noise sigma | 5.0 | [1.0, 15.0] | e- |
| Background | 50 | [10, 200] | counts |
| Dark current | 0.1 | [0.01, 1.0] | e-/px/s |

**Solvers**: VST+BM3D, CARE, Noise2Void. **Scenario I PSNR**: 18-25 dB. **Scenario II drop**: 3-10 dB.
**B3 True-Spec**: Alpha (87), read noise (4.2), background (63), dark current (0.15).
**B4 rho**: >= 0.70. **Improvement**: Add spatially-varying background, camera column noise.

---

### 1.3 Confocal Live-Cell (`confocal_livecell`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M1

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| PSF sigma | 1.5 | [0.8, 3.0] | px |
| Drift rate | 0.1 | [0, 1.0] | px/frame |
| Bleaching rate | 0.01 | [0, 0.1] | per frame |
| Pinhole misalignment | 0 | [0, 0.5] | AU offset |

**B3 True-Spec**: PSF, drift trajectory, bleaching curve, pinhole offset.
**B4 rho**: >= 0.75. **Improvement**: Add sample-induced aberration, compound drift+bleaching.

---

### 1.4 Confocal 3D Z-Stack (`confocal_3d`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M1

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Axial PSF sigma | 3.0 | [1.5, 6.0] | px |
| Refractive index | 1.515 | [1.33, 1.56] | - |
| Attenuation coeff | 0.03 | [0, 0.1] | per slice |
| Spherical aberration | 0 | [0, 0.5] | waves |

**B3 True-Spec**: RI, depth-dependent PSF, attenuation, aberration.
**B4 rho**: >= 0.70. **Improvement**: Add depth-dependent PSF model.

---

### 1.5 Structured Illumination Microscopy (`sim`)

**Canonical DAG**: M → C → D | **Carrier**: Photon | **Maturity**: M2

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Pattern frequency | 0.1 | [0.05, 0.15] | cycles/px |
| Phase shifts | [0, 2pi/3, 4pi/3] | +/- 0.2 rad each | rad |
| Modulation depth | 0.8 | [0.3, 1.0] | - |
| Pattern orientation | [0, 60, 120] | +/- 3 deg each | deg |

**Solvers**: Wiener-SIM, HiFi-SIM, fairSIM. **Scenario I PSNR**: 28-35 dB. **Scenario II drop**: 5-12 dB.
**B3 True-Spec**: Frequencies (3), phases (9), modulation depths (3), orientations (3), OTF.
**B4 rho**: >= 0.80. **Improvement**: Add 3D-SIM, nonlinear SIM, compound mismatch.

---

### 1.6 Light-Sheet LSFM (`lightsheet`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M1

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Sheet thickness | 5.0 | [2.0, 15.0] | um |
| Sheet tilt | 0 | [-3, 3] | deg |
| Stripe strength | 0.2 | [0, 0.8] | relative |
| Attenuation coeff | 0.02 | [0.005, 0.08] | per slice |

**B4 rho**: >= 0.75. **Improvement**: Add scattering-induced stripe, tile stitching error.

---

### 1.7 Fluorescence Lifetime FLIM (`flim`)

**Canonical DAG**: M → R → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| IRF width | 80 | [40, 200] | ps |
| IRF shift | 0 | [-50, 50] | ps |
| Afterpulsing | 0.01 | [0, 0.1] | relative |
| Pile-up fraction | 0 | [0, 0.05] | - |

**B4 rho**: >= 0.70. **Improvement**: Add multi-exponential, FRET efficiency benchmark.

---

### 1.8 Fourier Ptychographic FPM (`fpm`)

**Canonical DAG**: M → P → D | **Carrier**: Photon | **Maturity**: M1

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| LED position error | 0 | +/- 0.5 mm each | mm |
| LED intensity variation | 1.0 | [0.5, 1.5] per LED | relative |
| Pupil aberration (Zernike) | 0 | [0, 0.3] waves/mode | waves |
| Defocus | 0 | [-5, 5] | um |

**B4 rho**: >= 0.85. **Improvement**: Add vignetting, LED failure robustness.

---

### 1.9 Two-Photon (`two_photon`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Scattering coeff | 10 | [5, 30] | mm^-1 |
| PSF depth scaling | 1.0 | [0.7, 1.5] | - |
| Excitation attenuation | 0.01 | [0.005, 0.02] | per um |
| Motion artifact | 0 | [0, 5] | um |

**B4 rho**: >= 0.65. **Improvement**: Add adaptive optics, in-vivo motion model.

---

### 1.10 STED (`sted`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Depletion beam alignment | 0 | [0, 30] | nm offset |
| Saturation factor | 30 | [10, 50] | - |
| Effective PSF FWHM | 40 | [30, 120] | nm |

**B4 rho**: >= 0.70.

---

### 1.11 PALM/STORM (`palm_storm`)

**Canonical DAG**: M → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Drift rate (x, y) | 0 | [0, 2] each | nm/frame |
| Background photons | 20 | [5, 100] | per px |
| Photon count/event | 1000 | [200, 5000] | photons |
| Pixel size | 100 | [90, 110] | nm |

**B4 rho**: >= 0.80. **Improvement**: Add multi-emitter, 3D SMLM.

---

### 1.12 TIRF (`tirf`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Incidence angle | 68 | [62, 75] | deg |
| Evanescent depth | 100 | [50, 300] | nm |
| Background (non-TIRF) | 0 | [0, 0.3] | relative |

---

### 1.13 Polarization Microscopy (`polarization`)

**Canonical DAG**: M → C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Analyzer angle offset | 0 | [-5, 5] | deg |
| Retardance offset | 0 | [-10, 10] | nm |
| Extinction ratio | 1e-4 | [1e-5, 1e-3] | - |

---

### 1.14 Expansion Microscopy (`expansion`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Expansion factor | 4.0 | [3.5, 4.5] | x |
| Local distortion | 0 | [0, 5%] | relative |
| Anisotropic expansion | 0 | [0, 3%] | x vs y |

---

### 1.15 MINFLUX (`minflux`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Beam center error | 0 | [0, 5] | nm |
| Photon count | 500 | [50, 2000] | photons |

---

### 1.16 Image Scanning Microscopy (`ism`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Detector element offset | 0 | [-1, 1] | px |
| Magnification error | 0 | [-5%, 5%] | relative |

---

### 1.17 Phase Contrast Microscopy (`phase_contrast`) -- NEW

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Phase ring alignment | 0 | [0, 5] | um offset |
| Halo artifact strength | 0 | [0, 0.3] | relative |
| Phase ring absorption | 0.7 | [0.5, 0.9] | - |

**B3 True-Spec**: Phase ring position, absorption coefficient, condenser alignment.
**B4 rho**: >= 0.70. **Improvement**: Test quantitative phase recovery from phase contrast images.

---

### 1.18 Differential Interference Contrast (`dic`) -- NEW

**Canonical DAG**: M → C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Shear amount | 100 | [50, 200] | nm |
| Bias retardation | lambda/4 | +/- 30 nm | nm |
| Prism orientation | 0 | [-3, 3] | deg |

**B3 True-Spec**: Shear distance, bias, prism angle. **B4 rho**: >= 0.70.

---

### 1.19 Dark-Field Microscopy (`dark_field`) -- NEW

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Condenser NA vs objective NA ratio | 1.2 | [1.0, 1.5] | - |
| Stray light | 0 | [0, 5%] | relative |
| Scattering angle range | correct | +/- 10% | - |

---

### 1.20 Lattice Light-Sheet (`lattice_lightsheet`) -- NEW

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Lattice period error | 0 | [-5%, 5%] | relative |
| Dithering range | correct | +/- 10% | - |
| Sheet NA error | 0 | [-0.05, 0.05] | - |
| Excitation PSF sidelobe | 0 | [0, 10%] | relative |

**Improvement**: Compare Bessel vs lattice modes; add multi-view registration.

---

### 1.21 Second Harmonic Generation (`shg`) -- NEW

**Canonical DAG**: M → R → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Phase matching error | 0 | [-5%, 5%] | - |
| Excitation power fluctuation | 0 | [0, 10%] | - |
| Collection NA mismatch | 0 | [-0.1, 0.1] | - |

**B3 True-Spec**: Phase matching angle, excitation power, SHG efficiency. **B4 rho**: >= 0.65.

---

### 1.22 Spinning Disk Confocal (`spinning_disk`) -- NEW

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Pinhole crosstalk | 0 | [0, 15%] | - |
| Disk rotation wobble | 0 | [0, 1] | px |
| Illumination non-uniformity | 0 | [0, 10%] | - |

---

### 1.23 Three-Photon Microscopy (`three_photon`) -- NEW

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Scattering coeff | 15 | [8, 40] | mm^-1 |
| Excitation wavelength shift | 0 | [-10, 10] | nm |
| Depth-dependent PSF | varies | scale [0.5, 2.0] | - |

**Improvement**: Test imaging depth 1-2 mm in brain tissue; compare vs two-photon.

---

### 1.24 DNA-PAINT Super-Resolution (`dna_paint`) -- NEW

**Canonical DAG**: M → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Binding on-rate | varies | [0.5x, 2.0x] | relative |
| Imager strand concentration | 5 | [1, 20] | nM |
| Drift rate | 0 | [0, 3] | nm/frame |
| Background from non-specific binding | 0 | [0, 10%] | - |

**Improvement**: Add Exchange-PAINT (multi-target) benchmark.

---

## 2. Compressive Imaging (4 modalities)

*(Unchanged — CASSI, SPC, CACTI, Matrix with full detailed entries as before)*

---

### 2.1 CASSI (`cassi`)

**Canonical DAG**: M → W → Sigma → D | **Carrier**: Photon | **Maturity**: M3

#### B2: Mismatch Parameters (flagship-validated)
| Parameter | Nominal | Mismatch Range | True Example | Unit |
|-----------|---------|----------------|--------------|------|
| Mask shift dx | 0 | [-3.0, 3.0] | 1.47 | px |
| Mask shift dy | 0 | [-3.0, 3.0] | -0.23 | px |
| Mask rotation | 0 | [-2.0, 2.0] | 0.31 | deg |
| Dispersion slope a1 | 2.0 | [1.5, 2.5] | 2.01 | px/band |
| Dispersion offset alpha | 0 | [-0.5, 0.5] | 0.04 | px |
| Gain | 1.0 | [0.9, 1.1] | 1.02 | - |
| Read noise | 5.0 | [1.0, 15.0] | 5.1 | e- |

**Validated baseline**: GAP-TV +0.76 dB, rho = 0.85.
**B4 rho**: >= 0.85. **Improvement**: Compound all 5 params; Red Team adversarial; PSF spectral variation.

---

### 2.2 Single-Pixel Camera (`spc`)

**Canonical DAG**: M → Sigma → D | **Carrier**: Photon | **Maturity**: M3

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Gain drift alpha | 1.0 | [0.8, 1.2] | - |
| Measurement noise sigma_y | 0.01 | [0, 0.1] | - |
| Pattern error (bit flips) | 0 | [0, 1%] | - |

**Validated baseline**: FISTA-TV +7.71 dB, rho = 0.86.

---

### 2.3 CACTI (`cacti`)

**Canonical DAG**: M → Sigma → D | **Carrier**: Photon | **Maturity**: M3

#### B2: Mismatch Parameters (flagship-validated)
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Spatial shift x,y | 0 | [-3, 3] | px |
| Rotation | 0 | [-2, 2] | deg |
| Temporal clock error | 0 | [-0.5, 0.5] | frame frac |
| Gain / offset | 1.0 / 0 | [0.9,1.1] / [-5,5] | - / counts |
| Frame-dependent gain | 1.0 | [0.9, 1.1] per frame | - |

**Validated baseline**: GAP-TV +10.21 dB, rho = 1.00.

---

### 2.4 Generic Matrix Sensing (`matrix`)

**Canonical DAG**: M → D | **Carrier**: varies | **Maturity**: M1

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Matrix perturbation | 0 | [0, 10%] of ||A|| | - |
| Condition number change | kappa | [kappa, 10*kappa] | - |

---

## 3. Medical Imaging (37 modalities)

---

### 3.1 X-ray CT (`ct`)

**Canonical DAG**: Pi → D | **Carrier**: X-ray | **Maturity**: M3

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Center-of-rotation offset | 0 | [-5, 5] | px |
| Angular offset | 0 | [-3, 3] | deg |
| Detector tilt | 0 | [-2, 2] | deg |
| Beam hardening coeff | 0 | [0, 0.05] | - |
| Ring artifact amplitude | 0 | [0, 50] | counts |

**Validated baseline**: FBP +10.68 dB, rho = 1.00.
**Datasets**: Walnut Micro-CT, Helsinki Tomography Challenge 2022.
**Improvement**: Metal artifact reduction, limited-angle, scatter correction.

---

### 3.2 MRI (`mri`)

**Canonical DAG**: M → F → S → D | **Carrier**: Spin/RF | **Maturity**: M3

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Coil sensitivity error | 0 | [0, 15%] per coil | relative |
| k-space trajectory deviation | 0 | [0, 2%] | - |
| Off-resonance (B0) | 0 | [-100, 100] | Hz |
| Acceleration factor | R=4 | [2, 8] | - |

**Validated baseline**: SENSE +1.75 to +7.14 dB.
**Improvement**: Non-Cartesian, R=8/R=16, phase error estimation.

---

### 3.3 X-ray Radiography (`xray_radiography`)

**Canonical DAG**: Pi → D | **Carrier**: X-ray | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Scatter fraction | 0 | [0, 0.4] | - |
| Beam hardening | none | polynomial order 2-4 | - |
| Detector lag | 0 | [0, 0.1] | fraction |

---

### 3.4 Ultrasound B-mode (`ultrasound`)

**Canonical DAG**: P → D | **Carrier**: Acoustic | **Maturity**: M1

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Speed of sound | 1540 | [1450, 1600] | m/s |
| Phase aberration | 0 | [0, 50] | ns rms |
| Element sensitivity | 1.0 | [0.7, 1.3] per elem | - |
| Attenuation | 0.5 | [0.3, 0.8] | dB/cm/MHz |

**B4 rho**: >= 0.70. **Improvement**: Aberration correction, plane-wave ultrafast.

---

### 3.5 PET (`pet`)

**Canonical DAG**: Pi → D | **Carrier**: Gamma | **Maturity**: M1

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Attenuation map error | 0 | [0, 10%] | HU-to-LAC |
| Scatter fraction | 0.35 | [0.2, 0.5] | - |
| Randoms fraction | 0.2 | [0.1, 0.4] | - |
| Normalization error | 0 | [0, 5%] per det | - |
| TOF timing offset | 0 | [-200, 200] | ps |

**B4 rho**: >= 0.80. **Improvement**: PET/MR attenuation, motion correction.

---

### 3.6 SPECT (`spect`)

**Canonical DAG**: Pi → D | **Carrier**: Gamma | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Collimator response error | 0 | [0, 20%] FWHM | - |
| Center-of-rotation | 0 | [-3, 3] | px |
| Attenuation error | 0 | [0, 15%] | relative |

---

### 3.7-3.13 (Fluoroscopy, Mammography, DEXA, CBCT, Angiography, DOT, Photoacoustic)

*(Same mismatch tables as previous version — retained in full)*

**3.7 `fluoroscopy`**: Pi → D, X-ray. Lag [0,0.15], pincushion [0,3%], veiling glare [0,10%].

**3.8 `mammography`**: Pi → D, X-ray. Heel effect +/-10%, scatter-to-primary [0.2,0.8], MTF [0.8,1.2].

**3.9 `dexa`**: Pi → D, X-ray. Effective energies +/-15%, calibration +/-5%, fat fraction [0,20%].

**3.10 `cbct`**: Pi → D, X-ray. Scatter [0.2,0.7], truncation [0,20%], gantry flex [0,2] mm. rho >= 0.80.

**3.11 `angiography`**: Pi → D, X-ray. Motion [0,10] px, misregistration [-3,3] deg.

**3.12 `dot`**: M → R,P,R → D, Photon. mu_a [0.005,0.05], mu_s' [0.5,2.0], coupling [0.5,1.5].

**3.13 `photoacoustic`**: M → P → D, Acoustic. Speed [1400,1600] m/s, attenuation [0,0.5] dB/cm/MHz, fluence [0,30%].

---

### 3.14 OCT (`oct`)

**Canonical DAG**: P+P → Sigma → D | **Carrier**: Photon | **Maturity**: M1

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Dispersion GDD | 0 | [-100, 100] | fs^2 |
| Reference arm position | optimal | +/- 50 | um |
| K-linearization error | 0 | [0, 0.5%] | relative |

---

### 3.15-3.19 (fMRI, MRS, Diffusion MRI, Doppler US, Elastography)

**3.15 `fmri`**: M→F→S→D, Spin/RF. Distortion [0,5] px, dropout [0,15%], motion 6DOF [0,3] mm/deg.

**3.16 `mrs`**: M→F→S→D, Spin/RF. Lineshape, eddy phase [0,0.5] rad, residual water [0,100x].

**3.17 `diffusion_mri`**: M→F→S→D, Spin/RF. Gradient nonlinearity [0,5%], eddy distortion [0,3] px.

**3.18 `doppler_ultrasound`**: P→D, Acoustic. Flow angle [0,90] deg, PRF aliasing +/-20%, wall filter [20,200] Hz.

**3.19 `elastography`**: P→D, Acoustic. Speed error +/-20%, reflection [0,30%], dispersion [0,20%].

---

### 3.20-3.25 (Endoscopy, Fundus, OCTA, Proton Therapy, Brachytherapy, Portal Imaging)

**3.20 `endoscopy`**: M→C→D, Photon. Fiber transmission [0,15%], distortion [0,5%], cross-talk [0,10%].

**3.21 `fundus`**: C→D, Photon. Aberration [0,0.5] waves, illumination [0,30%].

**3.22 `octa`**: P+P→Sigma→D, Photon. Bulk motion [0,50] um, projection artifacts [0,0.3].

**3.23 `proton_therapy_img`**: Pi→D, Proton. Range uncertainty +/-3%, MCS error [0,10%].

**3.24 `brachytherapy_img`**: Pi→D, Gamma. Source position [0,3] mm, applicator [0,2] mm.

**3.25 `portal_imaging`**: Pi→D, MV X-ray. Gantry sag [0,3] mm, flex [0,5] mm, MLC error [-1,1] mm.

---

### 3.26 Photon-Counting Spectral CT (`spectral_ct`) -- NEW

**Canonical DAG**: Pi → W → D | **Carrier**: X-ray | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Energy threshold calibration | 0 | [-2, 2] | keV per bin |
| Charge sharing fraction | 0 | [0, 15%] | - |
| Pile-up at high flux | 0 | [0, 10%] | - |
| Material decomposition basis error | 0 | [0, 5%] | - |

**B3 True-Spec**: Energy thresholds, charge sharing model, pile-up model, basis functions.
**B4 rho**: >= 0.75. **Improvement**: K-edge subtraction benchmark, multi-material decomposition.

---

### 3.27 MR Elastography (`mr_elastography`) -- NEW

**Canonical DAG**: M → F → S → D | **Carrier**: Spin/RF | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Shear wave frequency error | 0 | [-10%, 10%] | - |
| Wave attenuation model | correct | +/- 20% | - |
| Motion encoding gradient error | 0 | [-5%, 5%] | - |
| Boundary reflection | none | [0, 20%] | amplitude |

---

### 3.28 CEST MRI (`cest_mri`) -- NEW

**Canonical DAG**: M → F → S → D | **Carrier**: Spin/RF | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| B0 inhomogeneity | 0 | [-50, 50] | Hz |
| B1 inhomogeneity | 0 | [0, 20%] | - |
| Saturation power error | 0 | [-10%, 10%] | - |
| MT contamination | 0 | [0, 30%] | - |

---

### 3.29 Arterial Spin Labeling (`asl_mri`) -- NEW

**Canonical DAG**: M → F → S → D | **Carrier**: Spin/RF | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Labeling efficiency | 0.85 | [0.6, 0.95] | - |
| Transit delay | 1.5 | [0.5, 3.0] | s |
| T1 blood error | 0 | [-10%, 10%] | - |

---

### 3.30 MR Angiography (`mra`) -- NEW

**Canonical DAG**: M → F → S → D | **Carrier**: Spin/RF | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Contrast timing error | 0 | [-3, 3] | s |
| Background suppression | complete | [0, 20%] residual | - |
| Velocity encoding error | 0 | [-15%, 15%] | - |

---

### 3.31 Susceptibility-Weighted Imaging (`swi`) -- NEW

**Canonical DAG**: M → F → S → D | **Carrier**: Spin/RF | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Phase unwrapping error | 0 | [0, 5%] of voxels | - |
| Background field removal error | 0 | [0, 10%] | - |
| Dipole inversion regularization | optimal | +/- 50% | - |

---

### 3.32 MR Fingerprinting (`mr_fingerprinting`) -- NEW

**Canonical DAG**: M → F → S → D | **Carrier**: Spin/RF | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Dictionary resolution (T1, T2) | fine | coarse [2x, 5x] | - |
| B1 inhomogeneity | 0 | [0, 15%] | - |
| Undersampling artifact | 0 | [0, 20%] | - |

---

### 3.33 Intravascular Ultrasound (`ivus`) -- NEW

**Canonical DAG**: P → D | **Carrier**: Acoustic | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Catheter rotation non-uniformity | 0 | [0, 10%] | - |
| Ring-down artifact | 0 | [0, 20%] depth | - |
| Sound speed in plaque | 1540 | [1400, 1700] | m/s |

---

### 3.34 Contrast-Enhanced Ultrasound (`ceus`) -- NEW

**Canonical DAG**: P → R → D | **Carrier**: Acoustic | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Bubble concentration | optimal | [0.1x, 5x] | relative |
| Nonlinear harmonic extraction | clean | [0, 10%] tissue leak | - |
| Motion between frames | 0 | [0, 5] | mm |

---

### 3.35 Digital Breast Tomosynthesis (`digital_breast_tomo`) -- NEW

**Canonical DAG**: Pi → D | **Carrier**: X-ray | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Angular range error | 0 | [-2, 2] | deg total |
| Detector motion blur | 0 | [0, 0.5] | px |
| Scatter fraction | 0.3 | [0.1, 0.6] | - |

---

### 3.36 Confocal Endomicroscopy (`confocal_endomicroscopy`) -- NEW

**Canonical DAG**: M → C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Fiber bundle honeycomb pattern | measured | +/- 5% pitch | - |
| Motion artifact | 0 | [0, 10] | px/frame |
| Fluorescein concentration variation | 1.0 | [0.3, 3.0] | relative |

---

### 3.37 Functional NIRS (`nirs_brain`) -- NEW

**Canonical DAG**: M → R,P → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Source-detector coupling | 1.0 | [0.5, 1.5] per optode | - |
| Scalp-brain distance variation | 0 | [0, 5] | mm |
| Motion artifact (head) | 0 | [0, 10%] signal | - |
| Systemic physiology contamination | 0 | [0, 30%] | - |

---

## 4. Coherent Imaging (5 modalities)

---

### 4.1 Ptychography (`ptychography`)

**Canonical DAG**: M → P → D | **Carrier**: Electron/Photon | **Maturity**: M3

**Validated baseline**: ePIE +7.09 dB, rho = 1.00.

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Probe position error | 0 | [-5, 5] each | px |
| Defocus | 0 | [-50, 50] | nm |
| Partial coherence | 1.0 | [0.7, 1.0] | - |

---

### 4.2 Holography (`holography`)

**Canonical DAG**: P → D | **Carrier**: Photon | **Maturity**: M1

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Reference angle error | 0 | [-1, 1] | deg |
| Carrier frequency error | 0 | [-5%, 5%] | - |
| Vibration | 0 | [0, lambda/10] | - |

---

### 4.3 Phase Retrieval / CDI (`phase_retrieval`)

**Canonical DAG**: P → D | **Carrier**: Photon/Electron | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Support mask error | 0 | [0, 10%] area | - |
| Oversampling ratio | 2.0 | [1.5, 4.0] | - |
| Partial coherence | 1.0 | [0.7, 1.0] | - |

---

### 4.4 Optical Diffraction Tomography (`odt`) -- NEW

**Canonical DAG**: P → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Illumination angle error | 0 | [-2, 2] | deg per angle |
| Missing cone artifact | 30 | [20, 50] | deg |
| Refractive index of medium | 1.337 | [1.33, 1.35] | - |
| Multiple scattering | none | [0, 10%] | - |

**B3 True-Spec**: Illumination angles, medium RI, sample 3D RI distribution.
**B4 rho**: >= 0.70. **Improvement**: Add Rytov vs Born approximation comparison.

---

### 4.5 Talbot-Lau X-ray Grating Interferometry (`talbot_lau`) -- NEW

**Canonical DAG**: M → P → D | **Carrier**: X-ray | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Grating alignment (rotation) | 0 | [-0.5, 0.5] | deg |
| Inter-grating distance error | 0 | [-1%, 1%] | - |
| Phase stepping error | 0 | [-5%, 5%] | per step |
| Grating defect fraction | 0 | [0, 3%] | - |

**B3 True-Spec**: Grating periods, distances, phase steps, defect map.
**B4 rho**: >= 0.75. **Improvement**: Simultaneous absorption/phase/dark-field reconstruction.

---

## 5. Computational Photography (5 modalities)

---

### 5.1 Lensless (`lensless`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M3. **Validated**: ADMM +3.55 dB, rho = 0.78.

#### B2: PSF shift [−5,5] px, scale [0.9,1.1], defocus +/−50 um, rotation [−2,2] deg.

---

### 5.2 Panorama (`panorama`)

**Canonical DAG**: C → D | **Carrier**: Photon | **Maturity**: M0.

#### B2: Focus distance +/−10%, registration [0,3] px.

---

### 5.3 Coded Exposure / Flutter Shutter (`coded_exposure`) -- NEW

**Canonical DAG**: M → C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Shutter code timing error | 0 | [-5%, 5%] per slot | - |
| Motion blur PSF mismatch | 0 | [0, 20%] | velocity error |
| Sensor readout noise | 5 | [1, 15] | e- |

**B3 True-Spec**: Exact shutter timing sequence, true motion velocity.

---

### 5.4 Event Camera / DVS (`event_camera`) -- NEW

**Canonical DAG**: M → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Contrast threshold | 0.3 | [0.1, 0.5] | log intensity |
| Refractory period | 1 | [0.1, 10] | us |
| Noise event rate | 0 | [0, 1%] | of real events |
| Hot pixel fraction | 0 | [0, 0.5%] | - |

**Improvement**: Test HDR reconstruction, high-speed video reconstruction from events.

---

### 5.5 HDR Imaging (`hdr_imaging`) -- NEW

**Canonical DAG**: M → Sigma → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Camera response function error | 0 | [0, 10%] | - |
| Exposure ratio error | 0 | [-10%, 10%] | - |
| Ghost artifact (motion between exposures) | 0 | [0, 5] | px |

---

## 6. Computational Optics (2 modalities)

### 6.1 `light_field`: C→S→D. Microlens pitch +/-2%, rotation [-1,1] deg, f-number +/-30%.
### 6.2 `integral`: C→S→D. Lens position [0,0.5] mm, distortion [0,3%].

---

## 7. Neural Rendering (2 modalities)

### 7.1 `nerf`: M→P→D. Pose error [0,0.05] scene units, rotation [0,3] deg, focal +/-5%. rho >= 0.80.
### 7.2 `gaussian_splatting`: M→P→D. SfM noise [0,0.1], init density [10k,1M], pose error [0,0.03].

---

## 8. Electron Microscopy (11 modalities)

---

### 8.1-8.8 (Existing — SEM, TEM, Electron Tomo, STEM, 4D-STEM, EBSD, EELS, Electron Holography)

**8.1 `sem`**: C→D, Electron. Astigmatism [0,50] nm, drift [0,1] nm/s, charging [0,500] V.

**8.2 `tem`**: C→D, Electron. Defocus [-1000,1000] nm, Cs [0.5,2.5] mm, astigmatism [0,100] nm.

**8.3 `electron_tomography`**: Pi→D, Electron. Tilt axis [-3,3] px, mag variation +/-2%, missing wedge [20,50] deg.

**8.4 `stem`**: S→D, Electron. Scan distortion [0,3] px, probe aberration [-50,50] nm.

**8.5 `electron_diffraction`**: M→P→D, Electron. Camera length +/-5%, beam center +/-5 px.

**8.6 `ebsd`**: R→D, Electron. Pattern center +/-2%, detector tilt [-1,1] deg.

**8.7 `eels`**: S→D, Electron. Energy drift [-2,2] eV, gain instability [0,5%].

**8.8 `electron_holography`**: P→D, Electron. Biprism drift +/-2%, fringe rotation [-1,1] deg.

---

### 8.9 Cryo-Electron Tomography (`cryo_et`) -- NEW

**Canonical DAG**: Pi → D | **Carrier**: Electron | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Tilt axis offset | 0 | [-3, 3] | px |
| Tilt angle accuracy | 0 | [-1, 1] | deg per tilt |
| Dose-induced shrinkage | 0 | [0, 10%] | - |
| CTF per-tilt variation | varies | +/- 0.5 um defocus | um |
| Missing wedge | 30 | [20, 50] | deg |

**B3 True-Spec**: Tilt axis, angles, defocus per tilt, shrinkage trajectory, ice thickness.
**Improvement**: Subtomogram averaging benchmark, SIRT vs WBP comparison.

---

### 8.10 FIB-SEM (`fib_sem`) -- NEW

**Canonical DAG**: S → C → D | **Carrier**: Electron + Ion | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Slice thickness variation | 0 | [0, 15%] | - |
| Curtaining artifact | 0 | [0, 0.3] | relative |
| Charging | 0 | [0, 300] | V |
| Drift between slices | 0 | [0, 5] | nm |

---

### 8.11 STEM-EDX Elemental Mapping (`edx_mapping`) -- NEW

**Canonical DAG**: M → R → D | **Carrier**: Electron | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Absorption correction error | 0 | [0, 15%] | - |
| Detector solid angle | measured | +/- 10% | sr |
| Peak overlap (spectral) | 0 | [0, 3] elements | - |
| Bremsstrahlung background | measured | +/- 20% | - |

---

## 9. Depth Imaging (5 modalities)

---

### 9.1 `tof_camera`: P→D. Multi-path [0,30%], phase wrap +/-1, temperature offset [-5,5] cm.
### 9.2 `lidar`: P→S→D. Timing jitter [0,0.5] ns, angular error [-0.1,0.1] deg.
### 9.3 `structured_light`: M→C→D. Gamma [1.5,2.5], extrinsics [0,1] mm/deg, defocus [0,3] px.

---

### 9.4 Photometric Stereo (`photometric_stereo`) -- NEW

**Canonical DAG**: M → C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Light direction error | 0 | [0, 5] | deg per source |
| Light intensity calibration | 1.0 | [0.8, 1.2] per source | - |
| Non-Lambertian surface fraction | 0 | [0, 30%] | - |
| Cast shadow fraction | 0 | [0, 15%] | of pixels |

**Improvement**: Add near-field photometric stereo, uncalibrated photometric stereo.

---

### 9.5 Flash LiDAR (`flash_lidar`) -- NEW

**Canonical DAG**: P → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| SPAD jitter | 0 | [0, 100] | ps |
| Ambient photon rate | 0 | [0, 10x signal] | - |
| Pile-up distortion | 0 | [0, 20%] | at high flux |
| Pixel cross-talk | 0 | [0, 5%] | - |

---

## 10. Remote Sensing (11 modalities)

---

### 10.1-10.8 (Existing — SAR, Sonar, Hyperspectral, Multispectral, GPR, Weather Radar, Radio Interferometry, Passive Microwave)

**10.1 `sar`**: F→D, RF. Velocity +/-1%, motion phase [0,pi/4] rad.
**10.2 `sonar`**: P→D, Acoustic. Speed +/-2%, multipath 1-3 paths.
**10.3 `hyperspectral_remote`**: M→W→Sigma→D. Smile [0,2] px, keystone [0,2] px, atmospheric +/-10%.
**10.4 `multispectral_sat`**: M→Sigma→D. Band registration [0,2] px, MTF +/-10%.
**10.5 `gpr`**: P→D, RF. Permittivity +/-20%, clutter [0,-10] dB.
**10.6 `weather_radar`**: P→R→D, RF. Clutter [-40,-15] dBZ, attenuation [0,10] dB/km.
**10.7 `radio_interferometry`**: F→S→D, RF. Baseline [0,1] cm, atmospheric phase [0,1] rad.
**10.8 `passive_microwave`**: Sigma→D, RF. Antenna pattern +/-5%, gain drift +/-1%.

---

### 10.9 InSAR (`insar`) -- NEW

**Canonical DAG**: F → S → D | **Carrier**: RF | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Phase unwrapping error | 0 | [0, 5%] of pixels | - |
| Baseline estimation error | 0 | [0, 1] | m |
| Atmospheric phase screen | 0 | [0, 1] | rad rms |
| Temporal decorrelation | 0 | [0, 0.3] | coherence loss |

**Improvement**: DInSAR for deformation, time-series InSAR (SBAS, PSI).

---

### 10.10 Polarimetric SAR (`polsar`) -- NEW

**Canonical DAG**: F → M → D | **Carrier**: RF | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Cross-talk between polarizations | 0 | [0, -25] | dB |
| Channel imbalance | 0 | [0, 1] | dB |
| Faraday rotation | 0 | [0, 5] | deg |

---

### 10.11 Ocean Color Remote Sensing (`ocean_color`) -- NEW

**Canonical DAG**: M → Sigma → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Atmospheric correction error | 0 | [0, 15%] | - |
| Sun glint contamination | 0 | [0, 20%] of pixels | - |
| Vicarious calibration offset | 0 | [-3%, 3%] per band | - |

---

## 11. Industrial Inspection (10 modalities)

---

### 11.1-11.8 (Existing)

**11.1 `industrial_ct`**: Pi→D. CoR [-5,5] px, scatter [0.1,0.6], beam hardening, ring artifacts.
**11.2 `xray_ndt`**: Pi→D. SDD +/-5%, geometric unsharpness [0,1] mm.
**11.3 `ultrasonic_phased_array`**: P→D. Velocity +/-3%, coupling [0.5,1.5], wedge +/-2 deg.
**11.4 `eddy_current`**: F→D. Lift-off [0,0.5] mm, conductivity +/-10%.
**11.5 `active_thermography`**: P→D. Emissivity [0,15%], heating uniformity [0.7,1.3].
**11.6 `terahertz`**: P→D. Water vapor [0,5] dB, etalon [0,100] GHz, RI +/-5%.
**11.7 `machine_vision`**: C→D. Illumination [0,20%], MTF [0.2,0.8], distortion [0,3%].
**11.8 `xrf_imaging`**: M→R→D. Matrix effects [0,20%], self-absorption [0,30%], dead time [0,10%].

---

### 11.9 Shearography (`shearography`) -- NEW

**Canonical DAG**: M → P → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Shearing amount error | 0 | [-10%, 10%] | - |
| Speckle decorrelation | 0 | [0, 0.3] | - |
| Loading non-uniformity | 0 | [0, 20%] | - |

---

### 11.10 Scanning Acoustic Microscopy (`acoustic_microscopy`) -- NEW

**Canonical DAG**: P → D | **Carrier**: Acoustic | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Coupling medium speed | 1480 | [1450, 1550] | m/s |
| Focus depth error | 0 | [-20, 20] | um |
| Lens aberration | 0 | [0, 0.2] | waves |
| Gate position error | 0 | [-5%, 5%] | - |

---

## 12. Scientific Instrumentation (12 modalities)

---

### 12.1-12.8 (Existing)

**12.1 `xray_crystallography`**: F→S→D. Absorption +/-10%, radiation damage [0,20%], mosaicity +/-50%.
**12.2 `saxs`**: R→D. Beam divergence [0.05,0.5] mrad, parasitic scatter [0,20%].
**12.3 `maldi_msi`**: S→D. Mass drift [-10,10] ppm, matrix inhomogeneity [0,30%].
**12.4 `atom_probe`**: S→D. Trajectory aberration [0,10%], local magnification [0,20%].
**12.5 `cryo_em`**: C→D. Defocus [-0.5,-5.0] um, Cs [2.0,3.5] mm, beam tilt [0,1] mrad, ice [20,100] nm. rho >= 0.85.
**12.6 `neutron_tomo`**: Pi→D. Beam spectrum +/-10%, scattering [0,15%], gamma [0,5%].
**12.7 `proton_radiography`**: Pi→D. MCS error [0,15%], energy loss +/-5%.
**12.8 `muon_tomo`**: Pi→D. Angular resolution [3,15] mrad, alignment [0,1] mm.

---

### 12.9 WAXS (`waxs`) -- NEW

**Canonical DAG**: R → D | **Carrier**: X-ray | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Detector distance error | 0 | [-1%, 1%] | - |
| Beam center error | 0 | [0, 3] | px |
| Polarization correction | 1.0 | [0.9, 1.0] | - |
| Air scatter background | 0 | [0, 5%] | - |

---

### 12.10 XRF Tomography (`xrf_tomo`) -- NEW

**Canonical DAG**: Pi → R → D | **Carrier**: X-ray | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Self-absorption correction | 0 | [0, 30%] | - |
| Rotation axis offset | 0 | [-3, 3] | px |
| Fluorescence yield error | 0 | [-10%, 10%] | - |
| Dead time at high count rate | 0 | [0, 10%] | - |

---

### 12.11 Neutron Diffraction (`neutron_diffraction`) -- NEW

**Canonical DAG**: R → S → D | **Carrier**: Neutron | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Wavelength calibration | 0 | [-0.1%, 0.1%] | - |
| Absorption correction | 0 | [0, 10%] | - |
| Texture/preferred orientation | none | [0, 20%] | - |
| TOF frame overlap | 0 | [0, 5%] of peaks | - |

---

### 12.12 Cathodoluminescence (`cathodoluminescence`) -- NEW

**Canonical DAG**: M → R → D | **Carrier**: Electron | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Beam current drift | 0 | [0, 5%] | - |
| Collection efficiency variation | 0 | [0, 20%] | spatial |
| Spectral calibration error | 0 | [-2, 2] | nm |
| Carbon contamination | 0 | [0, 10%] | signal loss |

---

## 13. Broader Experimental Science (11 modalities)

---

### 13.1-13.8 (Existing)

**13.1 `adaptive_optics`**: M→C→D. Residual wavefront [0,lambda/4], r0 [5,30] cm, wind [5,30] m/s.
**13.2 `seismic_tomo`**: P→D. Velocity +/-10%, source location [0,5] km.
**13.3 `gravitational_wave`**: P→Sigma→D. Calibration +/-5%, PSD +/-10%, glitch [0,1]/100s.
**13.4 `particle_calorimetry`**: R→Sigma→D. Inter-cal [0,3%], non-linearity [0,5%].
**13.5 `radio_astronomy`**: F→S→D. Antenna gain [0,5%], phase [0,10] deg, RFI [0,5%].
**13.6 `acoustic_emission`**: P→S→D. Velocity anisotropy [0,15%], coupling [0.5,1.5].
**13.7 `magnetic_particle`**: M→F→D. System function [0,10%], relaxation [0,20%].
**13.8 `impedance_tomo`**: M→D. Contact impedance [50,500] ohm, electrode position [0,5] mm.

---

### 13.9 Full-Waveform Inversion (`fwi`) -- NEW

**Canonical DAG**: P → D | **Carrier**: Seismic/Acoustic | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Starting velocity model error | 0 | [-15%, 15%] | - |
| Source wavelet error | 0 | [-10%, 10%] amplitude | - |
| Anelastic attenuation (Q) | infinite | [50, 500] | - |
| Source location error | 0 | [0, 100] | m |

**Improvement**: Multi-scale FWI, elastic (multi-parameter) inversion.

---

### 13.10 Ocean Acoustic Tomography (`ocean_acoustic_tomo`) -- NEW

**Canonical DAG**: P → D | **Carrier**: Acoustic | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Sound speed profile error | 0 | [-2%, 2%] | - |
| Multipath identification | correct | [0, 20%] misassigned | - |
| Source/receiver position | 0 | [0, 10] | m |
| Current velocity error | 0 | [-0.5, 0.5] | m/s |

---

### 13.11 Bioluminescence Tomography (`bioluminescence_tomo`) -- NEW

**Canonical DAG**: Src → R,P → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Optical property error (mu_a, mu_s') | 0 | [0, 20%] | relative |
| Source depth ambiguity | 0 | [0, 5] | mm |
| Autofluorescence background | 0 | [0, 30%] | - |

---

## 14. Spectroscopy & Spectral Imaging (8 modalities) -- NEW CATEGORY

---

### 14.1 Raman Imaging (`raman_imaging`)

**Canonical DAG**: M → R → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Spectral calibration shift | 0 | [-2, 2] | cm^-1 |
| Fluorescence background | 0 | [0, 10x Raman signal] | relative |
| Laser power fluctuation | 0 | [0, 5%] | - |
| Cosmic ray artifact | 0 | [0, 1%] of spectra | - |

**B1 Example**: "Design confocal Raman for pharmaceutical tablet: 785 nm, 10 um resolution, 100-3200 cm^-1."
**B3 True-Spec**: Spectral calibration, laser power, fluorescence model, cosmic ray locations.
**B4 rho**: >= 0.75. **Improvement**: Add SERS benchmark, baseline subtraction comparison.

---

### 14.2 CARS Microscopy (`cars`)

**Canonical DAG**: M → R → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Pump-Stokes frequency offset | 0 | [-5, 5] | cm^-1 |
| Non-resonant background | 0 | [0, 50%] of signal | - |
| Chirp mismatch | 0 | [0, 500] | fs^2 |

---

### 14.3 Stimulated Raman Scattering (`srs`)

**Canonical DAG**: M → R → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Lock-in phase error | 0 | [-10, 10] | deg |
| Cross-phase modulation | 0 | [0, 5%] | - |
| Laser intensity noise (RIN) | -150 | [-140, -160] | dBc/Hz |

---

### 14.4 FTIR Imaging (`ftir_imaging`)

**Canonical DAG**: M → Sigma → D | **Carrier**: IR photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Wavenumber calibration | 0 | [-2, 2] | cm^-1 |
| Water vapor absorption | 0 | [0, variable] | - |
| Detector nonlinearity | 0 | [0, 5%] | - |
| ATR crystal RI error | 0 | [-1%, 1%] | - |

---

### 14.5 LIBS Imaging (`libs`)

**Canonical DAG**: M → R → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Laser energy fluctuation | 0 | [0, 10%] | - |
| Matrix effect | 0 | [0, 30%] | - |
| Self-absorption correction | 0 | [0, 20%] | - |
| Crater-to-crater variation | 0 | [0, 15%] | - |

---

### 14.6 Brillouin Microscopy (`brillouin`)

**Canonical DAG**: M → R → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Brillouin shift calibration | 0 | [-50, 50] | MHz |
| VIPA FSR error | 0 | [-0.5%, 0.5%] | - |
| Elastic scattering leakage | 0 | [0, -30] dB | - |

---

### 14.7 SIMS Imaging (`sims`)

**Canonical DAG**: S → D | **Carrier**: Ion | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Mass calibration drift | 0 | [-5, 5] | ppm |
| Matrix effect (sputter yield) | 0 | [0, 50%] | - |
| Crater edge effect | 0 | [0, 10%] of area | - |
| Charging (insulating samples) | 0 | [0, 200] | V |

---

### 14.8 DESI Mass Spec Imaging (`desi`)

**Canonical DAG**: S → D | **Carrier**: Ion | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Spray angle error | 0 | [-5, 5] | deg |
| Solvent flow variation | 0 | [0, 15%] | - |
| Ion suppression (matrix effect) | 0 | [0, 50%] | - |
| Spatial resolution degradation | 0 | [0, 50%] | - |

---

## 15. Ultrafast Imaging (4 modalities) -- NEW CATEGORY

---

### 15.1 Streak Camera (`streak_camera`)

**Canonical DAG**: M → Sigma → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Sweep nonlinearity | 0 | [0, 5%] | - |
| Temporal resolution | 1 | [0.5, 5] | ps |
| Dynamic range saturation | 0 | [0, 10%] of pixels | - |
| Trigger jitter | 0 | [0, 10] | ps |

**B1 Example**: "Design streak camera system for fluorescence lifetime: 2 ps resolution, 500 ps window."
**Improvement**: Add synchroscan mode, compressed streak (CUP variant).

---

### 15.2 Pump-Probe Microscopy (`pump_probe`)

**Canonical DAG**: M → R → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Time-zero drift | 0 | [-100, 100] | fs |
| Pump power fluctuation | 0 | [0, 5%] | - |
| Chirp (GDD) | 0 | [-500, 500] | fs^2 |
| Spatial overlap error | 0 | [0, 20%] of beam | - |

---

### 15.3 Compressed Ultrafast Photography (`cup`)

**Canonical DAG**: M → Sigma → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| DMD encoding error | 0 | [0, 2%] bit flip | - |
| Streak sweep calibration | 0 | [-5%, 5%] | - |
| Temporal-spatial coupling | 0 | [0, 10%] | - |

**B1 Example**: "Design T-CUP for light-in-flight: 10 trillion fps, 256x256 spatial."

---

### 15.4 XFEL Serial Crystallography (`xfel_sfx`)

**Canonical DAG**: M → R → D | **Carrier**: X-ray | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Hit rate | 10% | [1%, 30%] | - |
| Indexing ambiguity | 0 | [0, 10%] of patterns | - |
| Partiality model error | 0 | [0, 20%] | - |
| Background from jet/carrier | 0 | [0, 30%] | - |

---

## 16. Quantum Imaging (3 modalities) -- NEW CATEGORY

---

### 16.1 Ghost Imaging (`ghost_imaging`)

**Canonical DAG**: M → Sigma → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Bucket detector efficiency | 1.0 | [0.5, 1.0] | - |
| Speckle correlation mismatch | 0 | [0, 10%] | - |
| Background counts | 0 | [0, 5%] of signal | - |
| Number of measurements | 10000 | [1000, 100000] | - |

**B1 Example**: "Design ghost imaging system: thermal source, DMD modulation, single-pixel bucket detector."
**Improvement**: Computational ghost imaging vs quantum ghost imaging comparison.

---

### 16.2 Quantum Illumination (`quantum_illumination`)

**Canonical DAG**: M → R → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Entanglement quality (concurrence) | 1.0 | [0.5, 1.0] | - |
| Background thermal noise | 0 | [0, 100] photons/mode | - |
| Detector dark count rate | 0 | [0, 1000] | Hz |
| Channel loss | 0 | [0, 30] | dB |

---

### 16.3 Entangled Photon Microscopy (`entangled_photon`)

**Canonical DAG**: M → R → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Pair generation rate | optimal | [0.1x, 10x] | - |
| Coincidence window | 1 | [0.1, 10] | ns |
| Accidental coincidence rate | 0 | [0, 20%] of real | - |
| Photon loss (per arm) | 0 | [0, 6] | dB |

---

## 17. Multi-Modal Fusion (6 modalities) -- NEW CATEGORY

---

### 17.1 PET/CT Fusion (`pet_ct`)

**Canonical DAG**: Pi → D (CT) + Pi → D (PET) → Fusion | **Carrier**: X-ray + Gamma | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| CT-PET registration error | 0 | [0, 3] | mm |
| Attenuation map from CT error | 0 | [0, 10%] | HU-to-LAC |
| Respiratory motion mismatch | 0 | [0, 15] | mm |
| CT contrast agent artifact | 0 | [0, 20%] | attenuation |

**B1 Example**: "Design PET/CT protocol for lung staging: low-dose CT, FDG-PET, 3-min beds, gated."
**Improvement**: Respiratory gating, metal artifact propagation to PET.

---

### 17.2 PET/MR Fusion (`pet_mr`)

**Canonical DAG**: Pi → D (PET) + M → F → S → D (MR) → Fusion | **Carrier**: Gamma + RF | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| MR-based attenuation error | 0 | [0, 15%] | - |
| Susceptibility artifact at air/tissue | 0 | [0, 5] | mm |
| Timing synchronization | 0 | [0, 100] | ms |
| Truncation (MR FOV < PET FOV) | 0 | [0, 20%] | of body |

---

### 17.3 SPECT/CT Fusion (`spect_ct`)

**Canonical DAG**: Pi → D (SPECT) + Pi → D (CT) → Fusion | **Carrier**: Gamma + X-ray | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Registration error | 0 | [0, 5] | mm |
| CT-based attenuation error | 0 | [0, 10%] | - |
| Scatter correction error | 0 | [0, 15%] | - |

---

### 17.4 US/MRI Fusion (`us_mri`)

**Canonical DAG**: P → D (US) + M → F → S → D (MR) → Fusion | **Carrier**: Acoustic + RF | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Registration error (deformable) | 0 | [0, 10] | mm |
| Probe pressure deformation | 0 | [0, 15] | mm |
| MR distortion | 0 | [0, 5] | mm |

---

### 17.5 CT + Fluorescence (FLIT) (`ct_fluorescence`)

**Canonical DAG**: Pi → D (CT) + M → R,P → D (FLI) → Fusion | **Carrier**: X-ray + Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Optical property assignment error | 0 | [0, 30%] | - |
| Autofluorescence | 0 | [0, 50%] of signal | - |
| Registration (CT to optical) | 0 | [0, 3] | mm |

---

### 17.6 Correlative Light-Electron Microscopy (`clem`)

**Canonical DAG**: C → D (LM) + C → D (EM) → Fusion | **Carrier**: Photon + Electron | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Registration error (LM to EM) | 0 | [0, 500] | nm |
| Sample deformation (fixation) | 0 | [0, 5%] | shrinkage |
| Fluorescence preservation | 100% | [30%, 100%] | - |

---

## 18. Scanning Probe Microscopy (4 modalities) -- NEW CATEGORY

---

### 18.1 Atomic Force Microscopy (`afm`)

**Canonical DAG**: S → D | **Carrier**: Mechanical | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Tip shape convolution | ideal | +/- 30% radius | - |
| Piezo nonlinearity | 0 | [0, 5%] | - |
| Thermal drift | 0 | [0, 1] | nm/s |
| Scanner hysteresis | 0 | [0, 10%] | - |

**B1 Example**: "Design AFM scan for semiconductor feature metrology: tapping mode, 1 um scan, 512 lines."
**B3 True-Spec**: True tip shape, piezo calibration, drift trajectory, hysteresis curve.
**B4 rho**: >= 0.75. **Improvement**: Add tip deconvolution benchmark, high-speed AFM.

---

### 18.2 Scanning Tunneling Microscopy (`stm`)

**Canonical DAG**: S → D | **Carrier**: Electron | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Tip electronic structure | ideal | variable LDOS | - |
| Piezo creep | 0 | [0, 5%] | - |
| Tunneling barrier height | 4.5 | [3.0, 6.0] | eV |
| Vibration amplitude | 0 | [0, 5] | pm |

---

### 18.3 Near-field Optical Microscopy (`nsom`)

**Canonical DAG**: M → C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Tip-sample distance | 10 | [5, 50] | nm |
| Aperture size error | 0 | [-20%, 20%] | - |
| Topographic coupling | 0 | [0, 30%] | - |
| Far-field background | 0 | [0, 20%] | - |

---

### 18.4 Magnetic Force Microscopy (`mfm`)

**Canonical DAG**: S → M → D | **Carrier**: Magnetic | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Lift height | 50 | [20, 200] | nm |
| Tip magnetization model | point dipole | +/- 30% moment | - |
| Electrostatic coupling | 0 | [0, 10%] | - |

---

## 19. Astronomy & Space Imaging (4 modalities) -- NEW CATEGORY

---

### 19.1 Coronagraphy (`coronagraphy`)

**Canonical DAG**: M → P → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Coronagraph mask centering | 0 | [0, 0.1] | lambda/D |
| Wavefront error (WFE) | 0 | [0, lambda/100] rms | - |
| Stellar leakage | 1e-6 | [1e-7, 1e-4] | contrast |
| Speckle lifetime | static | [0.1, 100] | s |

**B1 Example**: "Design stellar coronagraph for exoplanet imaging: Lyot stop, 1e-8 contrast, 3 lambda/D IWA."
**Improvement**: Post-processing comparison (ADI, SDI, RDI); wavefront sensing & control loop.

---

### 19.2 Lucky Imaging (`lucky_imaging`)

**Canonical DAG**: M → C → D | **Carrier**: Photon | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Fried parameter (r0) | 15 | [5, 25] | cm |
| Frame selection threshold | 10% | [1%, 50%] | - |
| Isoplanatic angle | 5 | [2, 10] | arcsec |
| Registration error | 0 | [0, 0.5] | px |

---

### 19.3 Event Horizon Telescope Imaging (`eht_imaging`)

**Canonical DAG**: F → S → D | **Carrier**: RF (mm-wave) | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Atmospheric opacity (tau) | 0.1 | [0.05, 0.5] | nepers |
| Station gain calibration | 0 | [0, 10%] per station | - |
| uv-coverage sparsity | sparse | varies by night | - |
| Interstellar scattering | 0 | [0, 10] | uas broadening |

**Improvement**: Test different regularizers (MEM, RML, CLEAN, PRIMO).

---

### 19.4 Solar Imaging (`solar_imaging`)

**Canonical DAG**: M → P → D | **Carrier**: Photon/EUV | **Maturity**: M0

#### B2: Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| PSF degradation (mirror aging) | 0 | [0, 20%] | - |
| Stray light | 0 | [0, 5%] | - |
| Flat-field error | 0 | [0, 3%] | - |
| Pointing jitter | 0 | [0, 1] | arcsec |

---

## 20. Summary: All 168 Modalities

| # | Category | Count | M0 | M1 | M2 | M3 | M4 |
|---|----------|:-----:|:--:|:--:|:--:|:--:|:--:|
| 1 | Microscopy | 24 | 20 | 3 | 1 | 0 | 0 |
| 2 | Compressive Imaging | 4 | 0 | 1 | 0 | 3 | 0 |
| 3 | Medical Imaging | 37 | 29 | 4 | 0 | 2 | 0 |
| 4 | Coherent Imaging | 5 | 3 | 1 | 0 | 1 | 0 |
| 5 | Computational Photography | 5 | 4 | 0 | 0 | 1 | 0 |
| 6 | Computational Optics | 2 | 2 | 0 | 0 | 0 | 0 |
| 7 | Neural Rendering | 2 | 1 | 1 | 0 | 0 | 0 |
| 8 | Electron Microscopy | 11 | 11 | 0 | 0 | 0 | 0 |
| 9 | Depth Imaging | 5 | 5 | 0 | 0 | 0 | 0 |
| 10 | Remote Sensing | 11 | 11 | 0 | 0 | 0 | 0 |
| 11 | Industrial Inspection | 10 | 10 | 0 | 0 | 0 | 0 |
| 12 | Scientific Instrumentation | 12 | 12 | 0 | 0 | 0 | 0 |
| 13 | Broader Experimental Science | 11 | 11 | 0 | 0 | 0 | 0 |
| 14 | **Spectroscopy & Spectral** | **8** | **8** | 0 | 0 | 0 | 0 |
| 15 | **Ultrafast Imaging** | **4** | **4** | 0 | 0 | 0 | 0 |
| 16 | **Quantum Imaging** | **3** | **3** | 0 | 0 | 0 | 0 |
| 17 | **Multi-Modal Fusion** | **6** | **6** | 0 | 0 | 0 | 0 |
| 18 | **Scanning Probe** | **4** | **4** | 0 | 0 | 0 | 0 |
| 19 | **Astronomy & Space** | **4** | **4** | 0 | 0 | 0 | 0 |
| | **TOTAL** | **168** | **148** | **10** | **1** | **7** | **0** |

### Category Breakdown by Canonical DAG Family

| DAG Family | Primitive Pattern | Modality Count | Example |
|------------|------------------|:--------------:|---------|
| Deconvolution | C → D | 32 | widefield, SEM, fundus, AFM |
| Tomography | Pi → D | 24 | CT, PET, SPECT, electron tomo |
| MRI family | M → F → S → D | 12 | MRI, fMRI, MRS, ASL, MRF |
| Spectral | M → W → Sigma → D | 5 | CASSI, hyperspectral |
| Compressive | M → Sigma → D | 8 | SPC, CACTI, ghost imaging |
| Ptychographic | M → P → D | 10 | ptychography, FPM, coronagraphy |
| Propagation | P → D | 22 | ultrasound, sonar, ToF, seismic |
| Scattering | M → R → D | 16 | Raman, CARS, FLIM, XRF, Brillouin |
| Interferometric | F → S → D | 8 | radio, crystallography, EHT |
| Fourier | F → D | 3 | SAR, eddy current |
| Scanning probe | S → D | 8 | AFM, STM, MALDI, atom probe |
| Multi-modal | Combined | 6 | PET/CT, PET/MR, CLEM |
| Other | Various | 14 | HDR, coded exposure, event camera |

### Primitive Usage Frequency

| Primitive | Symbol | Used in N modalities |
|-----------|:------:|:-------------------:|
| Detect | D | 168 (all) |
| Modulate | M | 98 |
| Convolve | C | 52 |
| Project | Pi | 32 |
| Propagate | P | 38 |
| Sample | S | 28 |
| Scatter | R | 24 |
| Encode | F | 20 |
| Accumulate | Sigma | 16 |
| Disperse | W | 6 |
| Source | Src | (all, implicit) |

---

## 21. Priority Queue: Next 15 Modalities to Validate

| Priority | Modality | Category | Current | Key Dataset |
|:--------:|----------|----------|:-------:|-------------|
| 1 | `ultrasound` | Medical | M1 | Plane-wave US benchmark |
| 2 | `pet` | Medical | M1 | NEMA phantom (Zenodo) |
| 3 | `cryo_em` | Scientific | M0 | EMPIAR benchmark |
| 4 | `sim` | Microscopy | M2 | BioSR + pattern metadata |
| 5 | `oct` | Medical | M1 | Retinal OCT + calibration |
| 6 | `sar` | Remote Sensing | M0 | Sentinel-1 SLC |
| 7 | `fpm` | Microscopy | M1 | LED array calibration |
| 8 | `holography` | Coherent | M1 | Off-axis + ref calibration |
| 9 | `industrial_ct` | Industrial | M0 | GE/Nikon industrial phantom |
| 10 | `hyperspectral_remote` | Remote Sensing | M0 | AVIRIS / EnMAP |
| 11 | `raman_imaging` | Spectroscopy | M0 | Pharma tablet Raman |
| 12 | `pet_ct` | Multi-Modal | M0 | NEMA + CT phantom |
| 13 | `afm` | Scanning Probe | M0 | Si calibration grating |
| 14 | `ghost_imaging` | Quantum | M0 | DMD + bucket detector |
| 15 | `streak_camera` | Ultrafast | M0 | Fluorescence lifetime |

---

## 22. References

1. PWM Flagship Paper — Typed Primitives and OperatorGraph IR
2. [PWM Targeting System](targeting_system.md) — LIP Arena specification
3. [PWM Imaging Modality Registry](imaging_modalities.md) — 64-modality registry
4. [PWM Modality Standards](modality_standards.md) — Detailed standards for 17 modalities
5. [PWM Operator Mode](operator_mode.md) — Calibration pipeline
6. [PWM Canonical Primitives](plan_canonical_primitives.md) — 10 canonical primitives
7. [PWM Medical Physicist Targets](pwm_medical_physicist_targets.md) — Architecture overview
8. [PWM Modality Mismatch Guide](modality_mismatch_guide.md) — Mismatch evidence for 64 modalities
