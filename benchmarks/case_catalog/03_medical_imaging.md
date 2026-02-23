# Category 3: Medical Imaging — Full Case Catalog

> 37 modalities, 148 system variants, 37,888 total test instances

---

## 3.1 X-ray Computed Tomography (CT)

### System Variants (6)

| # | Variant | DAG | Physical Elements | Key Difference |
|---|---------|-----|-------------------|----------------|
| 1 | **Parallel-beam CT** | Src(synchrotron) -> Pi(Radon_parallel) -> D(flat_panel) | Synchrotron source + parallel beam geometry | Simplest; FBP exact |
| 2 | **Fan-beam CT** | Src(Xray_tube_120kVp) -> Pi(Radon_fan) -> D(arc_detector_512ch) | Rotating X-ray tube + arc detector array | Standard clinical 2D |
| 3 | **Cone-beam CT (CBCT)** | Src(Xray_tube) -> Pi(Radon_cone) -> D(flat_panel_1024x768) | X-ray tube + 2D flat-panel detector | Dental, radiation therapy |
| 4 | **Helical/Spiral CT** | Src(Xray_tube) -> Pi(Radon_helical) -> D(multi_row_64ch) | Rotating tube + translating table + multi-row detector | Standard clinical volumetric |
| 5 | **Dual-Energy CT** | Src(Xray_tube_80_140kVp) -> Pi(Radon_DE) -> D(dual_layer) | Rapid kV-switching or dual-source or dual-layer | Material decomposition |
| 6 | **Photon-Counting CT** | Src(Xray_tube) -> Pi(Radon) -> D(CdTe_energy_bins_4) | CdTe/Si energy-resolving detector (2-8 bins) | K-edge imaging; no electronic noise |

### Sizes (4)

| # | Sinogram | Reconstruction | Note |
|---|----------|---------------|------|
| 1 | 180 x 256 | 256 x 256 | Small (simulation) |
| 2 | 360 x 512 | 512 x 512 | Standard clinical |
| 3 | 720 x 1024 | 512 x 512 | High-res |
| 4 | 1000 x 736 | 512 x 512 | LoDoPaB standard |

### Dose Levels / Noise (4): Full-dose, Quarter-dose, Tenth-dose, Ultra-low-dose

### Mismatch Parameters (5)

| # | Parameter | Nominal | Range | Unit |
|---|-----------|---------|-------|------|
| 1 | Center-of-rotation offset | 0 | [-5, 5] | px |
| 2 | Angular offset | 0 | [-3, 3] | deg |
| 3 | Detector tilt | 0 | [-2, 2] | deg |
| 4 | Beam hardening coeff | 0 | [0, 0.05] | - |
| 5 | Ring artifact amplitude | 0 | [0, 50] | counts |

### CT Case Count

| Benchmark | Formula | Count |
|-----------|---------|-------|
| B1 | 6 x 4 x 3 | **72** |
| B2 | 6 x 4 x 4 x 5 = | **480** |
| B3 | **480** |
| B4 | **480** |
| **CT Total** | | **1,512** |

### CT Data Sources

| Source | Label | URL |
|--------|-------|-----|
| LoDoPaB-CT (40K+ slices) | `WEB` | https://zenodo.org/record/3384092 |
| AAPM Mayo Low-Dose CT | `WEB` | https://www.aapm.org/grandchallenge/lowdosect/ |
| LDCT-and-Projection-Data (TCIA) | `WEB` | https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/ |
| 2DeteCT (experimental, 5000 slices) | `WEB` | https://www.nature.com/articles/s41597-023-02484-6 |
| Cone-beam PCCT (15 walnuts) | `WEB` | https://www.nature.com/articles/s41597-025-06246-4 |
| Shepp-Logan phantom | `GEN` | numpy analytical phantom |
| FORBILD phantom | `GEN` | Analytical head/thorax phantom |

---

## 3.2 Magnetic Resonance Imaging (MRI)

### System Variants (6)

| # | Variant | DAG | Physical Elements |
|---|---------|-----|-------------------|
| 1 | **Cartesian Single-Coil** | Src(RF_excitation) -> M(tissue_T1_T2) -> F(k-space_Cartesian) -> S(uniform_grid) -> D(ADC) | Single receive coil, uniform grid |
| 2 | **Cartesian Multi-Coil (SENSE)** | Src(RF) -> M(coil_sensitivity_8ch) -> F(k-space) -> S(undersampled) -> D(ADC_8ch) | 8-32 coil array + SENSE reconstruction |
| 3 | **Cartesian Multi-Coil (GRAPPA)** | Same as above | GRAPPA k-space interpolation |
| 4 | **Radial Trajectory** | Src(RF) -> M(coil) -> F(k-space_radial) -> S(golden_angle) -> D(ADC) | Radial spokes through k-space center |
| 5 | **Spiral Trajectory** | Src(RF) -> M(coil) -> F(k-space_spiral) -> S(Archimedean) -> D(ADC) | Archimedean spiral readout |
| 6 | **3D Volumetric** | Src(RF) -> M(coil_32ch) -> F(k-space_3D) -> S(undersampled_3D) -> D(ADC_32ch) | Full 3D k-space acquisition |

### Contrasts (5): T1w, T2w, FLAIR, PD, DWI
### Acceleration Factors (4): R=2, R=4, R=8, R=16
### Sizes: 256x256, 320x320, 384x384 (3)

### Mismatch Parameters (4)

| # | Parameter | Range | Unit |
|---|-----------|-------|------|
| 1 | Coil sensitivity error | [0, 15%] per coil | relative |
| 2 | k-space trajectory deviation | [0, 2%] | - |
| 3 | Off-resonance (B0) | [-100, 100] | Hz |
| 4 | Acceleration factor R | [2, 16] | - |

### MRI Case Count

| Benchmark | Formula | Count |
|-----------|---------|-------|
| B1 | 6 x 4 x 3 | **72** |
| B2 | 6 x 5 contrasts x 4 accel x 3 sizes x 5 mismatch = | **1,800** |
| B3 | **1,800** |
| B4 | **1,800** |
| **MRI Total** | | **5,472** |

### MRI Data Sources

| Source | Label | URL |
|--------|-------|-----|
| fastMRI knee + brain (raw k-space) | `WEB` | https://fastmri.med.nyu.edu/ |
| fastMRI Prostate | `WEB` | https://www.nature.com/articles/s41597-024-03252-w |
| Calgary-Campinas CC-359 | `WEB` | https://sites.google.com/view/calgary-campinas-dataset |
| IXI Dataset (T1, T2, PD, MRA, DTI) | `WEB` | https://brain-development.org/ixi-dataset/ |
| BrainWeb (simulated, known GT) | `SYN-WEB` | https://brainweb.bic.mni.mcgill.ca/ |

---

## 3.3-3.5 X-ray Radiography, Fluoroscopy, Mammography

### System Variants (per modality, 2 each): Standard, Digital (DR)
### Case Count: 2 x 3 x 4 x 5 = **120 per benchmark per modality**
### Combined for 3 modalities: **360 per benchmark**

---

## 3.6 Ultrasound B-mode

### System Variants (5)

| # | Variant | DAG | Physical Elements |
|---|---------|-----|-------------------|
| 1 | **Focused Transmit** | Src(piezo_array_128ch) -> P(wave_focused) -> D(128ch_ADC) | Linear array, single focal point per line |
| 2 | **Plane-Wave** | Src(piezo_array) -> P(wave_planar) -> D(128ch_ADC) | Full-field illumination; coherent compounding |
| 3 | **Diverging Wave** | Src(phased_array) -> P(wave_diverging) -> D(64ch_ADC) | Cardiac/abdominal ultrafast |
| 4 | **Synthetic Aperture** | Src(single_element) -> P(wave) -> D(128ch) | Maximum flexibility; slowest |
| 5 | **3D/4D Ultrasound** | Src(matrix_array) -> P(wave_3D) -> D(matrix_ADC) | Volumetric acquisition |

### Sizes: 256x256, 512x512 (2)
### Compounding angles: 1, 11, 31, 75 (4)

### Mismatch Parameters (4)

| # | Parameter | Range | Unit |
|---|-----------|-------|------|
| 1 | Speed of sound error | [-50, 50] | m/s (from 1540 m/s) |
| 2 | Element pitch error | [-5%, 5%] | relative |
| 3 | Sampling frequency offset | [-2%, 2%] | relative |
| 4 | Clutter level | [0, -20] | dB |

### Case Count: 5 x 2 x 4 x 4 x 5 = **800 per benchmark**

### Data Sources

| Source | Label | URL |
|--------|-------|-----|
| PICMUS (simulation + phantom + in-vivo) | `WEB` | https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/ |
| CUBDL (49 phantom + 25 in-vivo) | `WEB` | https://cubdl.jhu.edu/data/ |
| USTB datasets | `WEB` | https://www.ustb.no/ustb-datasets/ |
| BrEaST breast lesion | `WEB` | https://www.nature.com/articles/s41597-024-02984-z |

---

## 3.7 PET (Positron Emission Tomography)

### System Variants (5)

| # | Variant | DAG | Physical Elements |
|---|---------|-----|-------------------|
| 1 | **2D PET** | Src(annihilation_511keV) -> Pi(LOR_2D) -> D(BGO_scintillator) | Septa between detector rings |
| 2 | **3D PET** | Src(annihilation) -> Pi(LOR_3D) -> D(LYSO_scintillator) | No septa; all LOR pairs accepted |
| 3 | **TOF-PET** | Src(annihilation) -> Pi(LOR_TOF) -> D(LYSO_fast_timing) | 200-400 ps timing resolution |
| 4 | **List-mode PET** | Src(annihilation) -> Pi(LOR_listmode) -> D(LYSO) | Event-by-event storage |
| 5 | **Total-body PET** | Src(annihilation) -> Pi(LOR_extended_FOV) -> D(LYSO_2m) | >1m axial FOV; uExplorer/Quadra |

### Sizes: 128x128x63, 256x256x89, 400x400x109 (3)
### Count levels (4): Standard, Half-dose, Quarter-dose, Ultra-low (1/10)
### Case Count: 5 x 3 x 4 x 5 = **300 per benchmark**

### Data Sources

| Source | Label | URL |
|--------|-------|-----|
| NEMA IQ Phantom (Zenodo, SyneRBI) | `WEB` | https://zenodo.org/records/8404015 |
| autoPET (1014 PET/CT, TCIA) | `WEB` | https://autopet.grand-challenge.org/Dataset/ |
| UDPET Challenge (1447 scans) | `WEB` | https://papers.miccai.org/miccai-2025/0959-Paper0232.html |
| Generated OSEM from Shepp-Logan | `GEN` | STIR/SIRF framework |

---

## 3.8 SPECT

### System Variants (3): Parallel-hole, Fan-beam, Pinhole collimator
### Case Count: 3 x 3 x 4 x 5 = **180 per benchmark**

---

## 3.9-3.11 CBCT, DEXA, Digital Breast Tomosynthesis

### CBCT Variants (2): Standard CBCT, Half-fan CBCT
### DEXA Variants (2): Pencil-beam, Fan-beam
### DBT Variants (2): Step-and-shoot, Continuous rotation
### Combined Case Count: 6 x 3 x 4 x 5 = **360 per benchmark per group**

---

## 3.12 Optical Coherence Tomography (OCT)

### System Variants (4)

| # | Variant | DAG | Physical Elements |
|---|---------|-----|-------------------|
| 1 | **Time-domain OCT** | Src(SLD) -> P(Michelson_interferometer) + P(reference) -> Sigma(interference) -> D(photodiode) | Scanning reference mirror |
| 2 | **Spectral-domain OCT** | Src(SLD) -> P(interferometer) -> Sigma -> D(spectrometer_2048ch) | Fixed reference; spectrometer detection |
| 3 | **Swept-source OCT** | Src(swept_laser_1060nm) -> P(interferometer) -> Sigma -> D(balanced_photodetector) | Wavelength-swept laser |
| 4 | **Full-field OCT** | Src(LED) -> P(interferometer) -> Sigma -> D(2D_camera) | En-face imaging; no scanning |

### Sizes: 256x512, 512x1024 (2)
### Case Count: 4 x 2 x 4 x 5 = **160 per benchmark**

---

## 3.13-3.19 Functional MRI, MRS, Diffusion MRI, Doppler US, Elastography, Endoscopy, Fundus Camera

All share MRI or US DAG patterns with modality-specific contrasts.

### Combined Variants: 2-3 per modality (avg 2.5) x 7 modalities = 17.5 -> 18
### Case Count (each): ~120 per benchmark
### Combined: 7 x 120 = **840 per benchmark**

---

## 3.20-3.37 Remaining Medical Modalities

Includes: OCTA, Proton Therapy Imaging, Brachytherapy Imaging, Portal Imaging, Spectral CT, MR Elastography, CEST MRI, ASL MRI, MRA, SWI, MR Fingerprinting, IVUS, CEUS, Confocal Laser Endomicroscopy, fNIRS, Angiography, DOT, Photoacoustic.

### Per modality: avg 2 variants x 2 sizes x 4 noise x 5 mismatch = **80 per benchmark**
### 18 remaining modalities x 80 = **1,440 per benchmark**

---

## Category 3 Summary

| Modality Group | Modalities | Per-Benchmark Total | B1 | B2 | B3 | B4 | Grand Total |
|---------------|-----------|---------------------|----|----|----|----|-------------|
| CT | 1 | 480 | 72 | 480 | 480 | 480 | 1,512 |
| MRI | 1 | 1,800 | 72 | 1,800 | 1,800 | 1,800 | 5,472 |
| Radiography/Fluoro/Mammo | 3 | 360 | 72 | 360 | 360 | 360 | 1,152 |
| Ultrasound | 1 | 800 | 60 | 800 | 800 | 800 | 2,460 |
| PET | 1 | 300 | 60 | 300 | 300 | 300 | 960 |
| SPECT | 1 | 180 | 36 | 180 | 180 | 180 | 576 |
| CBCT/DEXA/DBT | 3 | 360 | 72 | 360 | 360 | 360 | 1,152 |
| OCT | 1 | 160 | 48 | 160 | 160 | 160 | 528 |
| fMRI/MRS/DiffMRI/etc | 7 | 840 | 168 | 840 | 840 | 840 | 2,688 |
| Remaining 18 | 18 | 1,440 | 432 | 1,440 | 1,440 | 1,440 | 4,752 |
| **TOTAL** | **37** | **6,720** | **1,092** | **6,720** | **6,720** | **6,720** | **21,252** |
