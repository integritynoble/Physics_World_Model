# Category 2: Microscopy — Full Case Catalog

> 24 modalities, 72 system variants, 18,432 total test instances

---

## 2.1 Widefield Fluorescence Microscopy

### System Variants (3)

| # | Variant | DAG | Physical Elements |
|---|---------|-----|-------------------|
| 1 | **Standard Widefield** | Src(LED_470nm) -> C(PSF_Airy, NA=1.3) -> D(sCMOS_16bit) | Objective 40x/1.3 oil + tube lens + sCMOS |
| 2 | **Low-Dose Widefield** | Src(LED_low_power) -> C(PSF_Airy) -> D(sCMOS_16bit) | Same optics, reduced illumination (10-500 photons/px) |
| 3 | **Deconvolution Widefield** | Src(LED) -> C(PSF_measured) -> D(sCMOS) + RL_deconv | Richardson-Lucy or CARE post-processing |

### Sizes: 128x128, 256x256, 512x512, 1024x1024 (4)
### Noise: Clean, Low, Medium, High (4)
### Mismatch params: PSF sigma (1.2-3.5 px), background (0-200), gain (0.85-1.15), flatfield (0-15%), bleaching (0-0.05/frame)

### Case Count: 3 variants x 4 sizes x 4 noise x 5 mismatch = **240 per benchmark**
### Data Source: BioSR (`WEB`, figshare), BBBC (`WEB`, Broad), DeconvolutionLab2 (`WEB`, EPFL), CARE (`WEB`, MPI-CBG)

---

## 2.2 Confocal Live-Cell Microscopy

### System Variants (3)

| # | Variant | DAG | Physical Elements |
|---|---------|-----|-------------------|
| 1 | **Point-scanning Confocal** | Src(laser_488nm) -> M(pinhole_1AU) -> C(PSF_confocal) -> D(PMT) | Galvo scanner + pinhole + PMT/HyD |
| 2 | **Spinning Disk Confocal** | Src(laser) -> M(Nipkow_disk) -> C(PSF_confocal) -> D(sCMOS) | Yokogawa CSU-X1 + sCMOS |
| 3 | **Airyscan** | Src(laser) -> M(32element_detector) -> C(PSF_0.2AU) -> D(GaAsP_array) | Zeiss Airyscan 32-element GaAsP detector array |

### Sizes: 256x256, 512x512, 1024x1024 (3)
### Case Count: 3 x 3 x 4 x 5 = **180 per benchmark**
### Data Source: CARE Tribolium (`WEB`), SR-CACO-2 (`WEB`, NeurIPS 2024), Cell Tracking Challenge (`WEB`)

---

## 2.3 Confocal 3D Z-Stack

### System Variants (2): Standard confocal 3D, Airyscan 3D
### Sizes: 128x128x32, 256x256x64, 512x512x128 (3)
### Case Count: 2 x 3 x 4 x 5 = **120 per benchmark**
### Data Source: CARE datasets (`WEB`), DeconvolutionLab2 (`WEB`)

---

## 2.4 Structured Illumination Microscopy (SIM)

### System Variants (7)

| # | Variant | DAG | Physical Elements | Key Difference |
|---|---------|-----|-------------------|----------------|
| 1 | **2D-SIM (linear)** | Src(laser) -> M(grating_pattern, 3ang x 3phase) -> C(PSF_NA1.49) -> D(sCMOS) | SLM or grating + 9 raw frames | 2x lateral resolution |
| 2 | **3D-SIM** | Src(laser) -> M(3D_pattern, 3ang x 5phase) -> C(PSF_3D) -> D(sCMOS) | Same + axial modulation; 15 frames | 2x lateral + 3x axial |
| 3 | **NL-SIM (nonlinear)** | Src(high_power_laser) -> M(saturated_pattern) -> C(PSF) -> D(sCMOS) | Photoswitchable fluorophores; >25 frames | >2x resolution (5-7 harmonics) |
| 4 | **Lattice SIM** | Src(laser) -> M(lattice_spot_pattern) -> C(PSF) -> D(sCMOS) | Lattice spots not gratings; faster | Zeiss Lattice SIM 3/5 |
| 5 | **LLS-SIM** | Src(Bessel_lattice) -> M(lattice_pattern) -> C(PSF_lightsheet) -> D(sCMOS) | Lattice light-sheet + SIM combined | ~120nm lateral, ~160nm axial |
| 6 | **Instant SIM (iSIM)** | Src(laser) -> M(microlens_array) -> C(PSF_reassigned) -> D(sCMOS) | Optical pixel reassignment (spinning disk) | Real-time; ~1.4x gain |
| 7 | **openSIM** | Src(LED) -> M(DMD_pattern) -> C(PSF) -> D(sCMOS) | Open-hardware; UC2 compatible | Low-cost |

### Sizes: 128x128, 256x256, 512x512, 1024x1024 (4)
### Mismatch params: Pattern freq (0.05-0.15), phase shifts (+/-0.2 rad), modulation depth (0.3-1.0), orientation (+/-3 deg)
### Case Count: 7 x 4 x 4 x 5 = **560 per benchmark**
### Data Source: BioSR (`WEB`, figshare), Open-3DSIM (`WEB`, figshare), fairSIM (`WEB`), BioSR for LLS-SIM (`WEB`, Zenodo)

---

## 2.5 Light-Sheet Fluorescence Microscopy (LSFM)

### System Variants (6)

| # | Variant | DAG | Physical Elements |
|---|---------|-----|-------------------|
| 1 | **SPIM (OpenSPIM)** | Src(laser_sheet) -> C(PSF_sheet, thickness=5um) -> D(sCMOS) | Cylindrical lens + detection objective (perpendicular) |
| 2 | **diSPIM** | [Src1->C->D1; Src2->C->D2] | Dual-view perpendicular light sheets |
| 3 | **Lattice Light-Sheet (LLSM)** | Src(Bessel_lattice) -> C(PSF_Bessel) -> D(sCMOS) | Bessel beam lattice; Janelia design |
| 4 | **OPM (Oblique Plane)** | Src(oblique_sheet_45deg) -> C(PSF) -> D(sCMOS) | Single-objective; 35-45 deg oblique |
| 5 | **mesoSPIM** | Src(laser_sheet_wide) -> C(PSF) -> D(sCMOS) | Large FOV; cleared-tissue |
| 6 | **ExA-SPIM** | Src(laser_sheet) -> C(PSF) -> D(sCMOS) | Expansion + SPIM; whole mouse brain |

### Sizes: 256x256x64, 512x512x128, 1024x1024x256 (3)
### Mismatch params: Sheet thickness (2-15um), sheet tilt (-3 to +3 deg), stripe strength (0-0.8), attenuation (0.005-0.08)
### Case Count: 6 x 3 x 4 x 5 = **360 per benchmark**
### Data Source: OpenSPIM (`WEB`), BioSR LLS-SIM (`WEB`, Zenodo), Cell Tracking Challenge (`WEB`)

---

## 2.6 Fluorescence Lifetime Imaging (FLIM)

### System Variants (3): Time-domain FLIM, Frequency-domain FLIM, Phasor FLIM
### Sizes: 256x256, 512x512 (2)
### Case Count: 3 x 2 x 4 x 5 = **120 per benchmark**
### Data Source: Generated (`GEN`) + FLIM-PAINT datasets if available

---

## 2.7 Fourier Ptychographic Microscopy (FPM)

### System Variants (4)

| # | Variant | DAG | Physical Elements |
|---|---------|-----|-------------------|
| 1 | **LED Array FPM** | Src(LED_array_15x15) -> M(thin_sample) -> P(objective_pupil) -> D(sCMOS) | 225 LEDs, 4mm pitch, 4x/0.1NA objective |
| 2 | **Dome Illumination FPM** | Src(LED_dome) -> M(sample) -> P(pupil) -> D(sCMOS) | Hemispherical LED dome |
| 3 | **3D FPM** | Src(LED_array) -> M(thick_sample_multislice) -> P(pupil) -> D(sCMOS) | Multi-slice forward model |
| 4 | **Multispectral FPM** | Src(RGB_LED_array) -> M(sample) -> P(pupil) -> D(sCMOS) | Color/spectral imaging via wavelength |

### Sizes: 256x256 (raw) -> 1024x1024 (HR), 512x512 -> 2048x2048 (2)
### Mismatch params: LED position error (mm), LED intensity variation (0.5-1.5), pupil aberration (0-0.3 waves), defocus (-5 to +5 um)
### Case Count: 4 x 2 x 4 x 5 = **160 per benchmark**
### Data Source: Waller Lab FPM (`WEB`, GitHub), BU CISL 3D FPM (`WEB`, GitHub), Colorful FPM (`WEB`, IEEE DataPort)

---

## 2.8 Two-Photon Microscopy

### System Variants (4): Standard 2PM, 2P Calcium Imaging, 2P+AO, 2P Endoscopy (GRIN)
### Sizes: 256x256, 512x512, 1024x1024 (3)
### Mismatch params: Scattering coeff (5-30 mm^-1), PSF depth scaling (0.7-1.5), attenuation (0.005-0.02/um), motion (0-5um)
### Case Count: 4 x 3 x 4 x 5 = **240 per benchmark**
### Data Source: Allen Brain Observatory (`WEB`), Neurofinder (`WEB`), STNeuroNet (`WEB`)

---

## 2.9 STED Microscopy

### System Variants (3): Standard STED, STED-FLIM, 3D-STED
### Sizes: 256x256, 512x512 (2)
### Mismatch params: Depletion beam alignment (0-30nm), saturation factor (10-50), effective PSF FWHM (30-120nm)
### Case Count: 3 x 2 x 4 x 5 = **120 per benchmark**
### Data Source: pySTED simulation (`WEB`, Nature MI 2024), Generated (`GEN`)

---

## 2.10 PALM/STORM Single-Molecule Localization

### System Variants (4): 2D-PALM, 2D-STORM, 3D-STORM (astigmatic), 3D-STORM (double-helix)
### Sizes: 64x64, 128x128, 256x256, 512x512 (4)
### Mismatch params: Drift (0-2 nm/frame), background (5-100 photons/px), photons/event (200-5000), pixel size (90-110nm)
### Case Count: 4 x 4 x 4 x 5 = **320 per benchmark**
### Data Source: SMLM Challenge 2016 (`WEB`, EPFL), SRM Hub (`WEB`, EPFL), DL-SMLM (`WEB`, Nature 2025)

---

## 2.11 TIRF Microscopy

### System Variants (2): Standard TIRF, Multi-angle TIRF
### Sizes: 256x256, 512x512 (2)
### Case Count: 2 x 2 x 4 x 5 = **80 per benchmark**

---

## 2.12 Polarization Microscopy

### System Variants (3): Linear Polarization, Mueller Matrix, Stokes Polarimetry
### Sizes: 256x256, 512x512 (2)
### Case Count: 3 x 2 x 4 x 5 = **120 per benchmark**
### Data Source: Generated (`GEN`)

---

## 2.13 Expansion Microscopy (ExM)

### System Variants (2): 4x Expansion, 10x Expansion (TenX)
### Case Count: 2 x 3 x 4 x 5 = **120 per benchmark**

---

## 2.14 MINFLUX Nanoscopy

### System Variants (2): 2D-MINFLUX, 3D-MINFLUX (pMINFLUX)
### Case Count: 2 x 2 x 4 x 5 = **80 per benchmark**
### Data Source: Generated (`GEN`)

---

## 2.15 Image Scanning Microscopy (ISM)

### System Variants (2): Standard ISM, Airyscan-ISM
### Case Count: 2 x 2 x 4 x 5 = **80 per benchmark**

---

## 2.16 Phase Contrast Microscopy

### System Variants (2): Zernike Phase Contrast, Quantitative Phase
### Case Count: 2 x 3 x 4 x 5 = **120 per benchmark**

---

## 2.17 Differential Interference Contrast (DIC)

### System Variants (2): Standard DIC, Quantitative DIC
### Case Count: 2 x 2 x 4 x 5 = **80 per benchmark**

---

## 2.18 Dark-Field Microscopy

### System Variants (2): Transmitted Dark-Field, Reflected Dark-Field
### Case Count: 2 x 2 x 4 x 5 = **80 per benchmark**

---

## 2.19 Lattice Light-Sheet Microscopy

### System Variants (3): Standard LLSM, dOPM, AO-LLSM
### Case Count: 3 x 3 x 4 x 5 = **180 per benchmark**
### Data Source: BioSR LLS-SIM (`WEB`, Zenodo), Dryad LS characterization (`WEB`)

---

## 2.20 Second Harmonic Generation (SHG) Microscopy

### System Variants (2): Standard SHG, Polarization-resolved SHG
### Case Count: 2 x 2 x 4 x 5 = **80 per benchmark**

---

## 2.21 Spinning Disk Confocal

### System Variants (2): Yokogawa CSU-X1, CSU-W1 (SoRa)
### Case Count: 2 x 3 x 4 x 5 = **120 per benchmark**
### Data Source: Cell Tracking Challenge (`WEB`)

---

## 2.22 Three-Photon Microscopy

### System Variants (2): 1300nm 3PM, 1700nm 3PM
### Case Count: 2 x 2 x 4 x 5 = **80 per benchmark**

---

## 2.23 DNA-PAINT Super-Resolution

### System Variants (3): Standard DNA-PAINT, Exchange-PAINT (multiplexed), DNA-PAINT MINFLUX
### Sizes: 64x64, 128x128, 256x256 (3)
### Mismatch params: Binding on-rate (0.5-2.0), imager concentration (1-20 nM), drift (0-3 nm/frame), non-specific background (0-10)
### Case Count: 3 x 3 x 4 x 5 = **180 per benchmark**
### Data Source: Zenodo 6563100 (`WEB`), NanTex SRM (`WEB`, Zenodo)

---

## 2.24 Correlative Light-Electron Microscopy (CLEM) — listed under Microscopy

(Covered in Multi-Modal Fusion section)

---

## Category 2 Summary

| # | Modality | Variants | Per-Benchmark Cases | B1 | B2 | B3 | B4 | Total |
|---|----------|----------|---------------------|----|----|----|----|-------|
| 1 | Widefield | 3 | 240 | 36 | 240 | 240 | 240 | 756 |
| 2 | Confocal Live-Cell | 3 | 180 | 36 | 180 | 180 | 180 | 576 |
| 3 | Confocal 3D | 2 | 120 | 24 | 120 | 120 | 120 | 384 |
| 4 | SIM | 7 | 560 | 84 | 560 | 560 | 560 | 1,764 |
| 5 | Light-Sheet | 6 | 360 | 72 | 360 | 360 | 360 | 1,152 |
| 6 | FLIM | 3 | 120 | 36 | 120 | 120 | 120 | 396 |
| 7 | FPM | 4 | 160 | 48 | 160 | 160 | 160 | 528 |
| 8 | Two-Photon | 4 | 240 | 48 | 240 | 240 | 240 | 768 |
| 9 | STED | 3 | 120 | 36 | 120 | 120 | 120 | 396 |
| 10 | PALM/STORM | 4 | 320 | 48 | 320 | 320 | 320 | 1,008 |
| 11 | TIRF | 2 | 80 | 24 | 80 | 80 | 80 | 264 |
| 12 | Polarization | 3 | 120 | 36 | 120 | 120 | 120 | 396 |
| 13 | Expansion | 2 | 120 | 24 | 120 | 120 | 120 | 384 |
| 14 | MINFLUX | 2 | 80 | 24 | 80 | 80 | 80 | 264 |
| 15 | ISM | 2 | 80 | 24 | 80 | 80 | 80 | 264 |
| 16 | Phase Contrast | 2 | 120 | 24 | 120 | 120 | 120 | 384 |
| 17 | DIC | 2 | 80 | 24 | 80 | 80 | 80 | 264 |
| 18 | Dark-Field | 2 | 80 | 24 | 80 | 80 | 80 | 264 |
| 19 | Lattice LS | 3 | 180 | 36 | 180 | 180 | 180 | 576 |
| 20 | SHG | 2 | 80 | 24 | 80 | 80 | 80 | 264 |
| 21 | Spinning Disk | 2 | 120 | 24 | 120 | 120 | 120 | 384 |
| 22 | Three-Photon | 2 | 80 | 24 | 80 | 80 | 80 | 264 |
| 23 | DNA-PAINT | 3 | 180 | 36 | 180 | 180 | 180 | 576 |
| | **TOTAL** | **66** | **3,720** | **816** | **3,720** | **3,720** | **3,720** | **11,976** |
